#!/bin/sh
# Runs on system resume (systemd sleep hook) and via chatbot-resume.service.
case "$1" in
  post|"")
    logger -t chatbot-resume "Wake/resume detected; scheduling recover"
    sleep 5
    su - jadaboawwad -c '/home/jadaboawwad/Files/Software/Repositories/Applications/Portfolio/chatbot/deploy/systemd/chatbot-recover.sh' &
    ;;
esac
exit 0
