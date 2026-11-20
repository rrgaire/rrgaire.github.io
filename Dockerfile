# Local development image for rrgaire.github.io
#
# Everything is pinned so the image builds the same way on any machine:
#   - Ruby: exact version + image digest below
#   - Gems: exact versions in Gemfile.lock (committed)
#
# To upgrade Ruby, change the FROM line AND `ruby-version` in .github/workflows/deploy.yml together,
# then regenerate Gemfile.lock (see LOCAL_WORKFLOW.md).
FROM ruby:3.4.7-trixie@sha256:779ca68b84fe3804e27f5b7ea5ab001005c708b7236a798c15b860d71cba9d21

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        imagemagick \
        inotify-tools \
        jupyter-nbconvert \
        locales \
        procps \
        zlib1g-dev \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    JEKYLL_ENV=development

WORKDIR /srv/jekyll

# Install the exact gem versions from Gemfile.lock (cached unless Gemfile/Gemfile.lock change)
COPY Gemfile Gemfile.lock ./
RUN gem install bundler -v "$(tail -n 1 Gemfile.lock | tr -d ' ')" --no-document \
    && bundle config set --global frozen true \
    && bundle install --jobs 4 --retry 3

COPY bin/entry_point.sh /usr/local/bin/entry_point.sh

EXPOSE 8080 35729

CMD ["/usr/local/bin/entry_point.sh"]
