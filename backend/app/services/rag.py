import asyncio
import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import Settings
from app.models.schemas import SourceChunk
from app.services.context_filter import apply_context_filters
from app.services.hybrid_search import (
    BM25Retriever,
    _chunk_key,
    reciprocal_rank_fusion,
    rescore_by_query_term_overlap,
)
from app.services.query_expansion import (
    CERT_INTENT_RE,
    INTRO_SEARCH_QUERY,
    PORTFOLIO_INTENT_SEARCH,
    PROJECTS_INTENT_RE,
    QueryExpander,
    SKILLS_INTENT_RE,
    extract_named_app_search_term,
    has_latin_tokens,
    is_company_experience_question,
    is_contact_question,
    is_plural_app_role_question,
    is_single_app_role_question,
)
from app.services.language import (
    needs_arabic_polish,
    normalize_phone_numbers,
    resolve_language,
    sanitize_arabic_answer,
    strip_empty_numbered_items,
    strip_meta_source_phrases,
)
from app.services.question_intent import (
    is_degraded_arabic_answer,
    is_general_info_question,
    is_greeting_question,
)
from app.services.reranker import Reranker
from app.services.vector_store import VectorStoreManager

_NO_KB_MENTION_RULE_AR = (
    "20. ممنوع في الإجابة ذكر «قاعدة المعرفة» أو «المعلومات المتوفرة» أو «المصادر» "
    "أو أنك تجلب أو تستند إلى معلومات؛ أجب مباشرة دون الإشارة إلى آلية الإجابة.\n"
)
_NO_KB_MENTION_RULE_EN = (
    "Never mention knowledge base, sources, or that you retrieved information; "
    "answer directly.\n"
)
_CONTEXT_LABEL_AR = "**معلومات السيرة:**"
_CONTEXT_LABEL_EN = "**Resume information:**"

ARABIC_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "أنت مساعد يجيب بالعربية الفصحى عن سيرة جهاد أبو عواد المهنية.\n\n"
            "القواعد:\n"
            "1. المصادر قد تكون بأي لغة، لكن الإجابة عربية فصحى بالكامل.\n"
            "2. ترجم المصطلحات والأسماء الأجنبية إلى العربية، لكن انسخ عناوين "
            "البريد الإلكتروني وروابط LinkedIn وGitHub والموقع الشخصي كاملة "
            "كما في المصدر (مثل: name@email.com أو https://...).\n"
            "3. لا تدمج حروفاً لاتينية داخل كلمات عربية.\n"
            "4. لا تنسخ أخطاء OCR. لا تضع مراجعاً أو أقواساً في الإجابة "
            "(المصادر تُعرض منفصلة في الواجهة).\n"
            "5. عند السرد استخدم قائمة واضحة؛ ضع كل بند مرقّماً في سطر مستقل "
            "يبدأ بـ ١. أو ٢. (رقم عربي ثم نقطة ثم النص في نفس السطر).\n"
            "6. للتعريف العام أو التحية: 3-4 جمل مهذبة فقط، دون سرد السيرة كاملة. "
            "لا تُعرّف نفسك كنموذج لغوي (مثل جِيما)؛ أنت مساعد للإجابة عن سيرة جهاد.\n"
            "7. للأسئلة المحددة: إجابة وافية بالتفاصيل ذات الصلة فقط "
            "(المسميات، الشركات، التواريخ، التقنيات) دون حشو.\n"
            "8. ترجم كل المصطلحات التقنية إلى العربية (React → رياكت، "
            "TypeScript → تايب سكريبت، Node.js → نود، GraphQL → جراف كيو إل، "
            "PostgreSQL → بوستجري). لا تترك كلمات إنجليزية ولا قوائم فارغة "
            "(ممنوع: «باستخدام، و، و» أو «مثل و و»).\n"
            "9. اكتب أرقام الهاتف بالتنسيق الدولي مع + في البداية "
            "(مثل: +962 77 700 2130) ولا تعكس ترتيب الأرقام.\n"
            "10. أجب مباشرة من المصادر دون شرح خطوات البحث أو التحليل أو الاستنتاج. "
            "إذا وُجدت الإجابة في المصادر فلا تقل إنها غير متوفرة.\n"
            "11. استخدم فقط المقاطع المرتبطة بالسؤال؛ لا تخلط بين أقسام أو جهات مختلفة "
            "إذا كان السؤال محدداً بجهة أو موضوع معيّن.\n"
            "12. إن لم تجد المعلومة في المصادر المقدمة فقط، قل بوضوح أنها غير متوفرة "
            "ولا تخمّن.\n"
            "13. عند السؤال عن قائمة (شهادات، مهارات، مشاريع، خبرات): اذكر كل عنصر "
            "وارداً في المصادر ذات الصلة مع اسمه وتفاصيله وتاريخه إن وُجد؛ لا تُلخّص "
            "في فئات عامة ولا تعتمد على المقدمة العامة إذا وُجد قسم تفصيلي في المصادر. "
            "استخدم ترقيماً عربياً متسلسلاً فقط (١. ٢. ٣.) بنداً لكل عنصر في سطر "
            "مستقل دون شرطات فرعية ودون مقدمة عامة.\n"
            "14. «كم بده/بكم/السعر/التكلفة» تعني التسعير والأتعاب وليس عدد المشاريع.\n"
            "15. عند استخدام قائمة مرقّمة أكمِل كل بند بجملة أو جملتين؛ "
            "لا تبدأ بنداً دون إنهائه ولا تتوقف في منتصف كلمة أو جملة. "
            "لا تبدأ رقماً جديداً (مثل ١١.) إلا إذا أكملت البند السابق بالكامل. "
            "ممنوع وضع رقم بند دون نص (مثل: ١٢. أو **١٢.**). "
            "لا تنتقل للإنجليزية ولا تنسخ المصدر حرفياً؛ أكمل القائمة حتى آخر عنصر "
            "في المصادر.\n"
            "16. عند السؤال عن المهارات: استخرجها من أقسام Software Skills و Hardware "
            "Skills في المصادر (لغات، أطر، تعلم آلي، عتاد، تحكم، VFD)؛ لا تذكر "
            "خبرات العمل أو التدريب إلا إذا سُئلت عنها صراحة. قسّم مهارات العتاد "
            "والتحكم إلى بنود مرقّمة منفصلة (تشخيص، أنظمة الدفع، التحكم، اللوحات).\n"
            "17. عند السؤال عن المشاريع: اذكر كل مشروع أو منتج وارداً في قسم الخبرة "
            "(اسم المشروع/الشركة، الهدف، التقنيات) بترقيم عربي متسلسل؛ ابدأ كل بند "
            "باسم المشروع أو الشركة كما في المصدر؛ لا تخلط مع المهارات أو الشهادات أو "
            "المقدمة العامة.\n"
            "18. عند السؤال عن دور جهاد في عدة تطبيقات أو منتجات (جمع): اذكر كل "
            "تطبيق وارداً في المصادر بترقيم عربي؛ بنداً واحداً لكل تطبيق/شركة.\n"
            "19. عند السؤال عن دور جهاد في تطبيق واحد محدد باسمه: أجب بفقرة أو فقرتين "
            "تفصيليتين وافيتين (ليس قائمة مرقّمة رفيعة)؛ اذكر الشركة أو المؤسسة، "
            "المسمى الوظيفي، الفترة إن وُجدت، المسؤوليات والمهام، والتقنيات "
            "المستخدمة من المصادر المتعلقة بهذا التطبيق فقط؛ لا تُجزّئ الدور إلى "
            "بنود مرقّمة قصيرة ولا تخلط مع تطبيقات أخرى.\n"
            + _NO_KB_MENTION_RULE_AR
        ),
        (
            "human",
            f"{_CONTEXT_LABEL_AR}\n{{context}}\n\n"
            "**السؤال:**\n{question}\n\n"
            "**الإجابة:**",
        ),
    ]
)

COMPANY_EXPERIENCE_ARABIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "أنت مساعد يجيب بالعربية الفصحى عن سيرة جهاد أبو عواد المهنية.\n\n"
            "السؤال عن خبرة جهاد في شركة أو مؤسسة محددة باسمها. أجب بفقرتين أو ثلاث "
            "فقرات تفصيلية متصلة (ممنوع استخدام قوائم مرقّمة أو ترقيم ١. ٢.).\n\n"
            "اذكر من المصادر كل ما يتوفر عن هذه الشركة تحديداً:\n"
            "- اسم الشركة والمسمى الوظيفي وفترة العمل\n"
            "- طبيعة المشروع أو المنتج والهدف منه\n"
            "- مسؤوليات جهاد ومهامه الفعلية\n"
            "- التقنيات والأدوات والمنهجيات المستخدمة\n"
            "- أي إنجازات أو تفاصيل إضافية واردة في المصادر\n\n"
            "لا تذكر شركات أو وظائف أخرى غير المطلوبة. لا تخلط مع WeFix أو Optimum "
            "أو غيرها إذا لم تكن هي الشركة المسؤولة عنها. ترجم المصطلحات التقنية إلى "
            "العربية. أجب مباشرة دون مقدمات عامة. "
            + _NO_KB_MENTION_RULE_AR.strip(),
        ),
        (
            "human",
            f"{_CONTEXT_LABEL_AR}\n{{context}}\n\n"
            "**السؤال:**\n{question}\n\n"
            "**الإجابة (فقرات تفصيلية دون ترقيم):**",
        ),
    ]
)

COMPANY_EXPERIENCE_ENGLISH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer about the person's experience at one specifically named company. "
            "Write two or three detailed paragraphs (no numbered lists). Include company "
            "name, job title, period, project purpose, responsibilities, tasks, and "
            "technologies about that company only. Do not mention other employers. "
            + _NO_KB_MENTION_RULE_EN.strip(),
        ),
        (
            "human",
            f"{_CONTEXT_LABEL_EN}\n{{context}}\n\n"
            "**Question:**\n{question}\n\n"
            "**Answer (detailed paragraphs, no numbering):**",
        ),
    ]
)

SINGLE_APP_ROLE_ARABIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "أنت مساعد يجيب بالعربية الفصحى عن سيرة جهاد أبو عواد المهنية.\n\n"
            "السؤال عن دور شخص في تطبيق واحد محدد باسمه. أجب بفقرتين أو ثلاث فقرات "
            "تفصيلية متصلة (ممنوع استخدام قوائم مرقّمة أو ترقيم ١. ٢.).\n\n"
            "اذكر من المصادر كل ما يتوفر عن هذا التطبيق تحديداً:\n"
            "- الشركة أو المؤسسة صاحبة العمل أو العميل\n"
            "- المسمى الوظيفي وفترة العمل إن وُجدت\n"
            "- طبيعة التطبيق وهدفه\n"
            "- مسؤوليات جهاد ومهامه الفعلية\n"
            "- التقنيات والأدوات المستخدمة\n"
            "- أي تفاصيل إضافية واردة في المصادر\n\n"
            "لا تذكر تطبيقات أخرى غير المطلوب. لا تكرر المعنى. "
            "ترجم المصطلحات التقنية إلى العربية. أجب مباشرة دون مقدمات عامة. "
            + _NO_KB_MENTION_RULE_AR.strip(),
        ),
        (
            "human",
            f"{_CONTEXT_LABEL_AR}\n{{context}}\n\n"
            "**السؤال:**\n{question}\n\n"
            "**الإجابة (فقرات تفصيلية دون ترقيم):**",
        ),
    ]
)

GREETING_ARABIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "أنت مساعد ذكي يجيب بالعربية الفصحى عن السيرة المهنية لجهاد أبو عواد.\n\n"
            "السؤال تحية أو طلب تعريف مختصر (مثل: مرحبا، السلام عليكم، من أنت؟).\n\n"
            "أجب بـ 3 إلى 4 جمل مهذبة فقط (ممنوع القوائم المرقّمة):\n"
            "- استخدم «وعليكم السلام ورحمة الله وبركاته» فقط إذا كانت رسالة المستخدم "
            "تحتوي «السلام عليكم» أو «سلام عليكم»؛ وإلا فابدأ بـ «أهلاً بك» أو «مرحباً».\n"
            "- عرّف نفسك كمساعد للإجابة عن سيرة جهاد (لا تقل إنك نموذج لغوي مثل جِيما).\n"
            "- لخّص من المصادر: اسمه، تخصصه (ميكاترونكس/إلكترونيات/مطور Full Stack) "
            "بجملة أو جملتين.\n"
            "- اختم بدعوة للسؤال عن الخبرات، المهارات، الشهادات، أو المشاريع.\n\n"
            "لا تسرد الوظائف أو المشاريع أو الشهادات ولا تذكر تفاصيل WeFix أو الشركات. "
            + _NO_KB_MENTION_RULE_AR.strip(),
        ),
        (
            "human",
            f"{_CONTEXT_LABEL_AR}\n{{context}}\n\n"
            "**السؤال:**\n{question}\n\n"
            "**الإجابة (فقرة قصيرة دون ترقيم):**",
        ),
    ]
)

GREETING_ENGLISH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a portfolio assistant for Jehad Abu Awwad's resume/CV.\n\n"
            "The user sent a greeting or asks who you are. Reply in 3-4 polite sentences "
            "(no numbered lists): welcome them in the same language as the question, "
            "briefly introduce Jehad (Mechatronics/Full Stack engineer), and invite "
            "questions about experience, skills, certifications, or projects. "
            "Do not list jobs or projects. " + _NO_KB_MENTION_RULE_EN.strip(),
        ),
        (
            "human",
            f"{_CONTEXT_LABEL_EN}\n{{context}}\n\n**Question:**\n{{question}}\n\n**Answer:**",
        ),
    ]
)

CONTACT_ARABIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "أنت مساعد يجيب بالعربية الفصحى عن سيرة جهاد أبو عواد المهنية.\n\n"
            "السؤال عن التواصل مع جهاد (مشروع، تعاون، توظيف، لدي مشروع، أو طلب رقم/بريد).\n\n"
            "أجب بفقرة أو فقرتين واضحتين (بدون قائمة مرقّمة) وتضمّن من المصادر:\n"
            "- البريد الإلكتروني كاملاً\n"
            "- رقم الهاتف بصيغة +962\n"
            "- رابط LinkedIn وGitHub والموقع الشخصي إن وُجدت\n\n"
            "إذا قال المستخدم أن لديه مشروع أو يريد التعاون على مشروع: اذكر وسائل "
            "التواصل وادعُه لمراسلة جهاد عبر البريد أو الهاتف لوصف المشروع والتفاصيل. "
            "لا تسرد وظائفه أو خبراته السابقة ولا تقترح التواصل عبر أماكن عمله السابقة.\n"
            "إذا سُئل عن الرقم أو البريد فقط: أجب مباشرة بالمعلومة المطلوبة. "
            + _NO_KB_MENTION_RULE_AR.strip(),
        ),
        (
            "human",
            f"{_CONTEXT_LABEL_AR}\n{{context}}\n\n"
            "**السؤال:**\n{question}\n\n"
            "**الإجابة:**",
        ),
    ]
)

CONTACT_ENGLISH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Give Jehad's contact details: email, phone (+962), LinkedIn, GitHub, website. "
            "For project inquiries, give contact info and invite them to email—do not list "
            "past job roles. " + _NO_KB_MENTION_RULE_EN.strip(),
        ),
        (
            "human",
            f"{_CONTEXT_LABEL_EN}\n{{context}}\n\n**Question:**\n{{question}}\n\n**Answer:**",
        ),
    ]
)

SINGLE_APP_ROLE_ENGLISH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer about the person's role in one specifically named application. "
            "Write two or three detailed paragraphs (no numbered lists). Include "
            "company/institution, job title, period, app purpose, responsibilities, "
            "tasks, and technologies. Do not mention other applications. "
            + _NO_KB_MENTION_RULE_EN.strip(),
        ),
        (
            "human",
            f"{_CONTEXT_LABEL_EN}\n{{context}}\n\n"
            "**Question:**\n{question}\n\n"
            "**Answer (detailed paragraphs, no numbering):**",
        ),
    ]
)

ARABIC_POLISH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "أنت محرر لغوي عربي. مهمتك إعادة صياغة مسودة إجابة لتصبح عربية فصحى "
            "سليمة نحوياً وبلاغياً.\n\n"
            "القواعد:\n"
            "1. التزم بحقائق المرجع والمسودة؛ لا تضف ولا تحذف معلومات.\n"
            "2. أزل كل الكلمات الإنجليزية واستبدلها بترجمة عربية كاملة "
            "(إلا: AI، ML، API، IoT، JWT). لا تترك حروف عطف دون كلمات.\n"
            "3. أصلح الأسماء والمصطلحات المختلطة أو المشوّهة.\n"
            "4. حسّن البنية: جمل واضحة، وقوائم منظمة عند الحاجة.\n"
            "5. أزل الأقواس الفارغة وأي مراجع بين قوسين.\n"
            "6. عند القوائم: احتفظ بكل عنصر من المسودة مع تفاصيله؛ "
            "لا تحذف بنوداً ولا تُبقِ التاريخ دون اسم.\n"
            "7. أعد النص النهائي فقط دون مقدمات أو تعليقات.",
        ),
        (
            "human",
            "**مرجع السيرة:**\n{context}\n\n"
            "**السؤال:**\n{question}\n\n"
            "**المسودة:**\n{draft}\n\n"
            "**الإجابة المنقّحة:**",
        ),
    ]
)

ENGLISH_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an intelligent assistant for Jehad Abu Awwad's professional profile.\n\n"
            "Important Instructions:\n"
            "1. Read the provided information carefully and use it to answer.\n"
            "2. Provide a concise, unified answer (at most two short paragraphs).\n"
            "3. If the information contains the answer, use it directly with appropriate context.\n"
            "4. If the information is partial, clearly state what is available.\n"
            "5. Provide ONE consolidated answer only. Do not repeat the same "
            "information in different wording.\n"
            "6. Answer directly from the sources. Do not describe your search process.\n"
            "7. If the answer is present in the sources, do not claim it is missing.\n"
            "8. Use only passages relevant to the question; do not mix unrelated sections.\n"
            "9. Only if the provided sources contain no relevant information, say: "
            '"I cannot find an answer in the provided information."\n'
            "10. When the question asks for a list (certifications, skills, projects, "
            "experience), include every relevant item from the sources with names, "
            "details, and dates when present; do not summarize from the intro alone "
            "if a detailed section exists in the sources.\n"
            "11. When the question asks about the person's role across multiple "
            "applications, list every named app from the experience section.\n"
            "13. When the question asks about the role in one specific named "
            "application, answer in one or two detailed paragraphs with company, "
            "job title, period, responsibilities, tasks, and technologies from "
            "relevant sources only; do not use a thin numbered list.\n"
            "12. When using a numbered list, every item must include text on the same "
            "line. Never output a bare number marker (e.g. 12. or **12.**) with no content.\n"
            + _NO_KB_MENTION_RULE_EN
        ),
        (
            "human",
            f"{_CONTEXT_LABEL_EN}\n{{context}}\n\n"
            "**Question:**\n{question}\n\n"
            "**Answer (one cohesive response, no repetition):**",
        ),
    ]
)

KB_NOT_FOUND_AR = "لا أتوفّر لدي معلومة كافية للإجابة على هذا السؤال."


def _word_set(text: str) -> set[str]:
    return {word for word in re.findall(r"\w+", text.lower()) if len(word) > 2}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def deduplicate_chunks(
    chunks: list[tuple[Document, float]],
    similarity_threshold: float = 0.65,
) -> list[tuple[Document, float]]:
    unique: list[tuple[Document, float]] = []
    for doc, score in chunks:
        words = _word_set(doc.page_content)
        if any(
            _jaccard_similarity(words, _word_set(existing.page_content))
            >= similarity_threshold
            for existing, _ in unique
        ):
            continue
        unique.append((doc, score))
    return unique


def _paragraph_prefix(text: str, words: int = 5) -> str:
    tokens = re.findall(r"\w+", text.lower())
    return " ".join(tokens[:words])


SKILLS_SECTION_HEADERS = (
    "## **technical skills**",
    "## **software skills**",
    "## **hardware skills**",
)

SKILLS_BODY_HINTS = (
    "ladder, classical logic",
    "diagnosis and repair of electronic systems",
)

SKILLS_QUESTION_RE = re.compile(r"مهار|skills", re.IGNORECASE)
PROJECTS_QUESTION_RE = re.compile(r"مشاريع|مشروع|projects", re.IGNORECASE)

EXPERIENCE_SECTION_HEADERS = (
    "## **experience**",
    "## **●",
    "## ●",
)

LIST_SECTION_HINTS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (
        re.compile(r"شهاد|ترخيص|certif|license", re.IGNORECASE),
        ("licenses & certifications",),
    ),
    (
        SKILLS_QUESTION_RE,
        SKILLS_SECTION_HEADERS,
    ),
    (
        PROJECTS_QUESTION_RE,
        EXPERIENCE_SECTION_HEADERS,
    ),
    (
        re.compile(
            r"تطبيقات|applications|مختلفة?|مختلف",
            re.IGNORECASE,
        ),
        EXPERIENCE_SECTION_HEADERS,
    ),
    (
        re.compile(r"تعليم|education", re.IGNORECASE),
        ("education",),
    ),
]

LIST_QUESTION_RE = re.compile(
    r"شهاد|ترخيص|certif|license|مهار|skills|مشاريع|مشروع|projects|"
    r"تطبيقات|applications",
    re.IGNORECASE,
)


def is_experience_list_question(question: str) -> bool:
    if (
        is_single_app_role_question(question)
        or is_contact_question(question)
        or is_company_experience_question(question)
    ):
        return False
    return bool(
        PROJECTS_QUESTION_RE.search(question)
        or is_plural_app_role_question(question)
    )


CONTACT_SECTION_HINTS = (
    "@",
    "+962",
    "linkedin.com",
    "outlook.com",
    "jehadabuawwad.com",
    "github.com",
)


def is_contact_section_chunk(content: str) -> bool:
    lowered = content.lower()
    return any(hint in lowered for hint in CONTACT_SECTION_HINTS)


def prioritize_contact_chunks(
    chunks: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    contact = [
        item for item in chunks if is_contact_section_chunk(item[0].page_content)
    ]
    if contact:
        return contact
    return chunks


def is_intro_section_chunk(content: str) -> bool:
    lowered = content.lower()
    if "experienced mechatronics" in lowered:
        return True
    if "jehad abu awwad" in lowered and "mechatronics" in lowered:
        return True
    if "full stack developer" in lowered and "code fellows" in lowered:
        return True
    return False


def prioritize_intro_chunks(
    chunks: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    intro = [
        item for item in chunks if is_intro_section_chunk(item[0].page_content)
    ]
    if intro:
        intro.sort(key=lambda item: -len(item[0].page_content))
        return intro
    non_experience = [
        item
        for item in chunks
        if not is_experience_section_chunk(item[0].page_content)
    ]
    return non_experience or chunks


def prioritize_chunks_for_named_app(
    chunks: list[tuple[Document, float]],
    question: str,
) -> list[tuple[Document, float]]:
    """Keep chunks that mention the named application; prefer focused hits."""
    term = extract_named_app_search_term(question)
    if not term:
        return chunks
    needle = term.lower()
    matching = [
        item for item in chunks if needle in item[0].page_content.lower()
    ]
    if not matching:
        return chunks
    matching.sort(
        key=lambda item: (
            0 if needle in item[0].page_content.lower()[:300] else 1,
            -item[0].page_content.lower().count(needle),
            -len(item[0].page_content),
        )
    )
    return matching

FULL_LIST_QUESTION_RE = re.compile(
    r"شهاد|ترخيص|certif|license|اسرد|اذكر|أذكر|كل|جميع|all|every|complete",
    re.IGNORECASE,
)


def portfolio_num_predict(settings: Settings, question: str) -> int:
    """Token budget for portfolio answers — lists and intros need the full tier."""
    if is_greeting_question(question):
        return settings.portfolio_llm_num_predict_short
    if (
        CERT_INTENT_RE.search(question)
        or is_plural_app_role_question(question)
        or is_general_info_question(question)
        or is_experience_list_question(question)
        or LIST_QUESTION_RE.search(question)
        or (
            FULL_LIST_QUESTION_RE.search(question)
            and LIST_QUESTION_RE.search(question)
        )
    ):
        return settings.portfolio_llm_num_predict
    if is_single_app_role_question(question) or is_company_experience_question(question):
        return settings.portfolio_llm_num_predict_medium
    return settings.portfolio_llm_num_predict_short


def is_skills_section_chunk(content: str) -> bool:
    lowered = content.lower()
    if any(header in lowered for header in SKILLS_SECTION_HEADERS):
        return True
    return any(hint in lowered for hint in SKILLS_BODY_HINTS)


def _skills_chunk_rank(content: str) -> int:
    lowered = content.lower()
    if "## **software skills**" in lowered:
        return 0
    if "ladder, classical logic" in lowered:
        return 1
    if "## **hardware skills**" in lowered:
        return 2
    if "## **technical skills**" in lowered:
        return 3
    return 4


def skills_chunks_complete(chunks: list[tuple[Document, float]]) -> bool:
    has_software = any(
        "## **software skills**" in doc.page_content.lower() for doc, _ in chunks
    )
    has_hardware = any(
        "ladder, classical logic" in doc.page_content.lower()
        or (
            "## **hardware skills**" in doc.page_content.lower()
            and len(doc.page_content) > 80
        )
        for doc, _ in chunks
    )
    return has_software and has_hardware


def certs_chunks_complete(chunks: list[tuple[Document, float]]) -> bool:
    return any(
        "licenses & certifications" in doc.page_content.lower() for doc, _ in chunks
    )


def is_experience_section_chunk(content: str) -> bool:
    lowered = content.lower()
    if any(header in lowered for header in EXPERIENCE_SECTION_HEADERS):
        return True
    if "teams and projects" in lowered:
        return True
    if re.search(
        r"(?:internship|full time|part time).{0,40}(?:developer|engineer)",
        lowered,
    ):
        return True
    return False


def _experience_chunk_rank(content: str) -> tuple[int, int]:
    lowered = content.lower()
    if "## **experience**" in lowered:
        return (0, -len(content))
    if "teams and projects" in lowered:
        return (1, -len(content))
    if "## **●" in lowered or "## ●" in lowered:
        return (2, -len(content))
    return (3, -len(content))


def experience_chunks_complete(chunks: list[tuple[Document, float]]) -> bool:
    experience = [
        doc for doc, _ in chunks if is_experience_section_chunk(doc.page_content)
    ]
    unique_snippets = {doc.page_content[:220] for doc in experience}
    return len(unique_snippets) >= 5


def _compact_section_chunk(content: str, max_chars: int = 800) -> str:
    """Trim long KB chunks so list-style answers can include more distinct items."""
    text = content.strip()
    if len(text) <= max_chars:
        return text

    teams_match = re.search(r"teams and projects", text, re.IGNORECASE)
    if teams_match:
        header_lines: list[str] = []
        for line in text.splitlines():
            header_lines.append(line)
            if line.strip().startswith("##"):
                break
        teams_text = text[teams_match.start() :]
        next_header = re.search(r"\n##\s", teams_text[1:])
        if next_header:
            teams_text = teams_text[: next_header.start() + 1]
        combined = "\n".join(header_lines) + "\n" + teams_text.strip()
        if len(combined) <= max_chars:
            return combined
        return combined[:max_chars].rstrip() + "..."

    kept: list[str] = []
    for line in text.splitlines():
        if kept and re.match(r"^[A-Za-z].*:\s*$", line.strip()):
            break
        if (
            kept
            and line.strip().startswith("- ")
            and sum(1 for existing in kept if existing.strip().startswith("- ")) >= 1
        ):
            break
        kept.append(line)
        if len("\n".join(kept)) >= max_chars:
            break

    result = "\n".join(kept).strip()
    if len(result) > max_chars:
        result = result[:max_chars].rstrip() + "..."
    elif len(result) < len(text):
        result += "..."
    return result


def _compact_app_role_chunk(content: str, max_chars: int = 900) -> str:
    """One job/app entry: header + summary; keep Optimum teams-and-projects list."""
    text = content.strip()
    if len(text) <= max_chars:
        return text
    if re.search(r"teams and projects", text, re.IGNORECASE):
        return _compact_section_chunk(text, max_chars)

    kept: list[str] = []
    past_header = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("##"):
            kept.append(line)
            past_header = True
            continue
        if not past_header:
            kept.append(line)
            continue
        if stripped.startswith("- "):
            break
        if re.match(r"^[A-Za-z].*:\s*$", stripped):
            break
        kept.append(line)
        if len("\n".join(kept)) >= max_chars:
            break

    result = "\n".join(kept).strip()
    if len(result) > max_chars:
        result = result[:max_chars].rstrip() + "..."
    elif len(result) < len(text):
        result += "..."
    return result


def deduplicate_by_content_prefix(
    chunks: list[tuple[Document, float]],
    prefix_len: int = 220,
) -> list[tuple[Document, float]]:
    seen: set[str] = set()
    unique: list[tuple[Document, float]] = []
    for item in chunks:
        key = item[0].page_content[:prefix_len]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


async def augment_section_chunks(
    question: str,
    chunks: list[tuple[Document, float]],
    *,
    vector_manager,
    bm25: BM25Retriever,
    document_id: str | None,
    fetch_k: int = 20,
) -> list[tuple[Document, float]]:
    """Fetch missing KB section chunks for list-style questions (domain-agnostic)."""
    boost_specs: list[
        tuple[re.Pattern[str], callable[[list[tuple[Document, float]]], bool]]
    ] = [
        (CERT_INTENT_RE, certs_chunks_complete),
        (SKILLS_INTENT_RE, skills_chunks_complete),
        (PROJECTS_INTENT_RE, experience_chunks_complete),
        (re.compile(r"تطبيقات|applications|مختلف", re.IGNORECASE), experience_chunks_complete),
    ]

    merged = list(chunks)
    seen = {_chunk_key(doc) for doc, _ in merged}

    if is_greeting_question(question) and not any(
        is_intro_section_chunk(doc.page_content) for doc, _ in merged
    ):
        vector_hits = await asyncio.to_thread(
            vector_manager.search, INTRO_SEARCH_QUERY, fetch_k, document_id
        )
        bm25_hits = await bm25.search(
            INTRO_SEARCH_QUERY, k=fetch_k, document_id=document_id
        )
        for doc, score in vector_hits + bm25_hits:
            key = _chunk_key(doc)
            if key in seen:
                continue
            seen.add(key)
            merged.append((doc, score))

    if is_single_app_role_question(question) or is_company_experience_question(question):
        term = extract_named_app_search_term(question)
        if term and not any(
            term.lower() in doc.page_content.lower() for doc, _ in merged
        ):
            if is_company_experience_question(question):
                query = (
                    f"{term} experience job engineer responsibilities "
                    "technologies company project"
                )
            else:
                query = f"{term} role experience application mobile web developed"
            vector_hits = await asyncio.to_thread(
                vector_manager.search, query, fetch_k, document_id
            )
            bm25_hits = await bm25.search(query, k=fetch_k, document_id=document_id)
            for doc, score in vector_hits + bm25_hits:
                key = _chunk_key(doc)
                if key in seen:
                    continue
                seen.add(key)
                merged.append((doc, score))

    for pattern, is_complete in boost_specs:
        if not pattern.search(question) or is_complete(merged):
            continue
        query = next((q for p, q in PORTFOLIO_INTENT_SEARCH if p is pattern), None)
        if not query:
            continue
        vector_hits = await asyncio.to_thread(
            vector_manager.search, query, fetch_k, document_id
        )
        bm25_hits = await bm25.search(query, k=fetch_k, document_id=document_id)
        for doc, score in vector_hits + bm25_hits:
            key = _chunk_key(doc)
            if key in seen:
                continue
            seen.add(key)
            merged.append((doc, score))

    return merged


def prioritize_section_chunks(
    chunks: list[tuple[Document, float]],
    question: str,
) -> list[tuple[Document, float]]:
    """Put KB section chunks first for list-style questions (domain-agnostic)."""
    if SKILLS_QUESTION_RE.search(question):
        primary = [
            item for item in chunks if is_skills_section_chunk(item[0].page_content)
        ]
        if primary:
            primary.sort(key=lambda item: _skills_chunk_rank(item[0].page_content))
            return primary

    if is_experience_list_question(question):
        primary = [
            item
            for item in chunks
            if is_experience_section_chunk(item[0].page_content)
        ]
        if primary:
            primary.sort(
                key=lambda item: _experience_chunk_rank(item[0].page_content)
            )
            return primary

    markers: list[str] = []
    for pattern, section_markers in LIST_SECTION_HINTS:
        if pattern.search(question):
            markers.extend(section_markers)
    if not markers:
        return chunks

    primary = [
        item
        for item in chunks
        if any(marker in item[0].page_content.lower() for marker in markers)
    ]
    if not primary:
        return chunks

    if "licenses & certifications" in markers:
        primary.sort(
            key=lambda item: item[0].page_content.lower().count("certification"),
            reverse=True,
        )
        return primary

    return primary


def deduplicate_answer(text: str, similarity_threshold: float = 0.45) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [part.strip() for part in text.split("\n") if part.strip()]

    unique: list[str] = []
    seen_prefixes: set[str] = set()
    for paragraph in paragraphs:
        prefix = _paragraph_prefix(paragraph)
        words = _word_set(paragraph)
        if prefix in seen_prefixes:
            continue
        if any(
            _jaccard_similarity(words, _word_set(kept)) >= similarity_threshold
            for kept in unique
        ):
            continue
        seen_prefixes.add(prefix)
        unique.append(paragraph)

    return "\n\n".join(unique)


class RetrievalService:
    def __init__(
        self,
        vector_manager: VectorStoreManager,
        settings: Settings,
        reranker: Reranker | None = None,
        bm25: BM25Retriever | None = None,
        expander: QueryExpander | None = None,
    ) -> None:
        self.vector_manager = vector_manager
        self.settings = settings
        self.reranker = reranker
        self.bm25 = bm25 or BM25Retriever(settings)
        self.expander = expander or QueryExpander(settings)

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
        use_reranker: bool | None = None,
        use_hybrid: bool | None = None,
        portfolio_fast: bool = False,
    ) -> list[tuple[Document, float]]:
        k = top_k or self.settings.top_k
        min_k = (
            self.settings.portfolio_retrieval_min_k
            if portfolio_fast
            else self.settings.retrieval_min_k
        )
        fetch_multiplier = (
            self.settings.portfolio_retrieval_fetch_multiplier
            if portfolio_fast
            else self.settings.retrieval_fetch_multiplier
        )
        retrieval_k = max(k, min_k)

        should_rerank = (
            use_reranker
            if use_reranker is not None
            else self.settings.reranker_enabled
        )
        should_hybrid = (
            use_hybrid
            if use_hybrid is not None
            else self.settings.hybrid_search_enabled
        )

        fetch_k = (
            retrieval_k * fetch_multiplier
            if should_rerank and self.reranker
            else retrieval_k * 2
        )

        # Stage 1: multi-query expansion + hybrid retrieval + RRF fusion
        search_queries = await self.expander.expand(query, portfolio_fast=portfolio_fast)
        ranked_lists: list[list[tuple[Document, float]]] = []

        for search_query in search_queries:
            ranked_lists.append(
                await asyncio.to_thread(
                    self.vector_manager.search,
                    search_query,
                    fetch_k,
                    document_id,
                )
            )

        if should_hybrid:
            for bm25_query in self.expander.lexical_queries(search_queries):
                bm25_results = await self.bm25.search(
                    bm25_query, k=fetch_k, document_id=document_id
                )
                if bm25_results:
                    ranked_lists.append(bm25_results)

        if len(ranked_lists) == 1:
            candidates = ranked_lists[0][:fetch_k]
        else:
            candidates = reciprocal_rank_fusion(
                ranked_lists, fetch_k, self.settings.rrf_k
            )

        candidates = rescore_by_query_term_overlap(candidates, search_queries)

        # Stage 2: cross-encoder reranking on candidate pool
        if should_rerank and self.reranker and candidates:
            ranked = await asyncio.to_thread(
                self.reranker.rerank, query, candidates, k
            )
        else:
            ranked = candidates[:k]

        # Stage 3: relevance filtering before generation
        return apply_context_filters(
            ranked, self.settings, portfolio_fast=portfolio_fast
        )


class GenerationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Gemma 4 defaults to thinking mode; with num_predict capped it can
        # exhaust the budget on reasoning and return empty content.
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_num_predict,
            num_ctx=4096,
        ).bind(think=False)
        self.portfolio_llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_predict=settings.portfolio_llm_num_predict,
            num_ctx=8192,
        ).bind(think=False)
        self.portfolio_llm_medium = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_predict=settings.portfolio_llm_num_predict_medium,
            num_ctx=8192,
        ).bind(think=False)
        self.portfolio_llm_short = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_predict=settings.portfolio_llm_num_predict_short,
            num_ctx=8192,
        ).bind(think=False)

    def _portfolio_llm_for(self, question: str):
        tier = portfolio_num_predict(self.settings, question)
        if tier == self.settings.portfolio_llm_num_predict:
            return self.portfolio_llm
        if tier == self.settings.portfolio_llm_num_predict_medium:
            return self.portfolio_llm_medium
        return self.portfolio_llm_short

    def build_context(
        self,
        chunks: list[tuple[Document, float]],
        language: str = "en",
        portfolio_fast: bool = False,
        *,
        chunk_max_chars: int | None = None,
        total_max_chars: int | None = None,
        compact_chunks: bool = False,
        app_role_compact: bool = False,
    ) -> str:
        parts: list[str] = []
        max_chars = chunk_max_chars or (
            self.settings.portfolio_context_chunk_max_chars
            if portfolio_fast
            else self.settings.context_chunk_max_chars
        )
        max_total = total_max_chars
        if max_total is None and portfolio_fast:
            max_total = self.settings.portfolio_context_max_chars
        total_chars = 0
        for index, (doc, _) in enumerate(chunks, start=1):
            source = doc.metadata.get("filename", "unknown")
            page = doc.metadata.get("page")

            if language == "ar":
                header = f"[مصدر {index}: {source}]"
            else:
                header = f"[Source {index}: {source}]"

            if page:
                header += f" (page {page})" if language != "ar" else f" (صفحة {page})"

            content = doc.page_content.strip()
            if app_role_compact:
                content = _compact_app_role_chunk(content, max_chars)
            elif compact_chunks:
                content = _compact_section_chunk(content, max_chars)
            elif len(content) > max_chars:
                content = content[:max_chars].rstrip() + "..."
            block = f"{header}\n{content}"
            if max_total is not None and total_chars + len(block) > max_total:
                break
            parts.append(block)
            total_chars += len(block)

        return "\n\n---\n\n".join(parts)

    def to_source_chunks(
        self, chunks: list[tuple[Document, float]]
    ) -> list[SourceChunk]:
        sources: list[SourceChunk] = []
        for doc, score in chunks:
            sources.append(
                SourceChunk(
                    chunk_id=doc.id or "",
                    document_id=doc.metadata.get("document_id", ""),
                    filename=doc.metadata.get("filename", ""),
                    page=doc.metadata.get("page"),
                    score=round(float(score), 4),
                    text=doc.page_content[:500],
                )
            )
        return sources

    async def _polish_arabic(
        self, draft: str, question: str, context: str
    ) -> str:
        chain = ARABIC_POLISH_PROMPT | self.llm
        response = await chain.ainvoke(
            {"draft": draft, "question": question, "context": context}
        )
        return response.content

    async def generate(
        self,
        question: str,
        chunks: list[tuple[Document, float]],
        language: str | None = None,
        portfolio_fast: bool = False,
    ) -> str:
        lang = resolve_language(question, language)

        max_chunks = (
            self.settings.portfolio_max_context_chunks
            if portfolio_fast
            else self.settings.max_context_chunks
        )
        is_certs_list = bool(
            portfolio_fast
            and re.search(r"شهاد|ترخيص|certif|license", question, re.IGNORECASE)
        )
        is_greeting = bool(portfolio_fast and is_greeting_question(question))
        is_contact = bool(portfolio_fast and is_contact_question(question))
        is_single_app = bool(portfolio_fast and is_single_app_role_question(question))
        is_company_exp = bool(
            portfolio_fast and is_company_experience_question(question)
        )
        is_plural_app_role = bool(
            portfolio_fast and is_plural_app_role_question(question)
        )
        is_experience_list = bool(
            portfolio_fast and is_experience_list_question(question)
        )
        if is_greeting:
            merged = deduplicate_by_content_prefix(chunks)
            all_distinct = prioritize_intro_chunks(merged)[:2]
        elif is_contact:
            merged = deduplicate_by_content_prefix(chunks)
            all_distinct = prioritize_contact_chunks(merged)[:3]
        elif is_single_app or is_company_exp:
            merged = deduplicate_by_content_prefix(chunks)
            all_distinct = prioritize_chunks_for_named_app(merged, question)[:3]
        elif is_certs_list or is_experience_list:
            merged = deduplicate_by_content_prefix(chunks)
            all_distinct = prioritize_section_chunks(merged, question)[:max_chunks]
        else:
            all_distinct = deduplicate_chunks(chunks)[:max_chunks]
            if portfolio_fast:
                all_distinct = prioritize_section_chunks(all_distinct, question)
        context = self.build_context(
            all_distinct,
            language=lang,
            portfolio_fast=portfolio_fast,
            chunk_max_chars=(
                1200
                if is_greeting
                else (
                    1500
                    if is_contact
                    else (
                        2500
                        if is_single_app or is_company_exp
                        else (
                            900
                            if is_plural_app_role
                            else (1100 if is_experience_list else None)
                        )
                    )
                )
            ),
            total_max_chars=(
                2000
                if is_greeting
                else (
                    3000
                    if is_contact
                    else (
                        5500
                        if is_single_app or is_company_exp
                        else (8500 if is_experience_list else None)
                    )
                )
            ),
            compact_chunks=is_experience_list and not is_plural_app_role,
            app_role_compact=is_plural_app_role,
        )

        llm = (
            self._portfolio_llm_for(question)
            if portfolio_fast
            else self.llm
        )
        if is_greeting and lang == "ar":
            prompt = GREETING_ARABIC_PROMPT
        elif is_greeting:
            prompt = GREETING_ENGLISH_PROMPT
        elif is_contact and lang == "ar":
            prompt = CONTACT_ARABIC_PROMPT
        elif is_contact:
            prompt = CONTACT_ENGLISH_PROMPT
        elif is_company_exp and lang == "ar":
            prompt = COMPANY_EXPERIENCE_ARABIC_PROMPT
        elif is_company_exp:
            prompt = COMPANY_EXPERIENCE_ENGLISH_PROMPT
        elif is_single_app and lang == "ar":
            prompt = SINGLE_APP_ROLE_ARABIC_PROMPT
        elif is_single_app:
            prompt = SINGLE_APP_ROLE_ENGLISH_PROMPT
        else:
            prompt = ARABIC_RAG_PROMPT if lang == "ar" else ENGLISH_RAG_PROMPT
        chain = prompt | llm
        response = await chain.ainvoke({"context": context, "question": question})
        raw_answer = response.content or ""
        is_list_question = bool(
            LIST_QUESTION_RE.search(question)
            and not is_single_app_role_question(question)
            and not is_company_experience_question(question)
            and not is_contact_question(question)
            and not is_greeting_question(question)
        )
        answer = strip_empty_numbered_items(
            raw_answer if is_list_question else deduplicate_answer(raw_answer)
        )
        if lang == "ar" and answer:
            if self.settings.arabic_polish_enabled and (
                needs_arabic_polish(answer) or is_degraded_arabic_answer(answer)
            ):
                answer = await self._polish_arabic(answer, question, context)
            answer = sanitize_arabic_answer(
                answer,
                light=(
                    is_list_question
                    or is_single_app
                    or is_company_exp
                    or is_contact
                    or is_greeting
                ),
            )
            if is_degraded_arabic_answer(answer) and not is_greeting:
                answer = KB_NOT_FOUND_AR
        elif answer:
            answer = strip_meta_source_phrases(
                strip_empty_numbered_items(normalize_phone_numbers(answer))
            )
        return answer
