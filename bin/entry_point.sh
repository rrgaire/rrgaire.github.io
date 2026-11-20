#!/bin/bash
# Container entry point: serve the site with live reload, restart Jekyll when _config.yml changes.
set -uo pipefail

cd /srv/jekyll
CONFIG_FILE=_config.yml

start_jekyll() {
    bundle exec jekyll serve \
        --host=0.0.0.0 --port=8080 \
        --livereload --watch --force_polling \
        --destination /tmp/_site &
    JEKYLL_PID=$!
}

echo "Ruby: $(ruby -v)"
if ! bundle check >/dev/null 2>&1; then
    echo "Gemfile changed since the image was built; rebuild with: docker compose up -d --build"
    exit 1
fi

start_jekyll

# Jekyll does not reload _config.yml by itself, so restart it when the file changes.
while true; do
    if inotifywait -q -e modify,move,create,delete "$CONFIG_FILE"; then
        echo "Change detected to $CONFIG_FILE, restarting Jekyll"
        kill -INT "$JEKYLL_PID" 2>/dev/null   # graceful stop (no SIGTERM backtrace)
        wait "$JEKYLL_PID" 2>/dev/null
        start_jekyll
    else
        sleep 2
    fi
done
