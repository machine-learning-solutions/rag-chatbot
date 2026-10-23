# Match docker-compose pin.
FROM ollama/ollama:0.40.9

WORKDIR /app

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 11434

ENTRYPOINT ["/app/start.sh"]
