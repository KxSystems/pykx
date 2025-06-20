{#-
  This file was automatically generated - do not edit
-#}
<!-- Announcement begins-->
{% block announce %}
<div class="md-grid">
  <h2>New Documentation Site!</h2>
  <p style="font-size: 14px;">
      We are excited to announce the launch of our enhanced product documentation site for <a href="https://docs.kx.com/3.1/PyKX/home.htm" style="color: var(--md-typeset-a-color);">PyKX</a> at <a href="https://docs.kx.com/home/index.htm" style="color: var(--md-typeset-a-color);">docs.kx.com</a>. 
      It offers improved search capabilities, organized navigation, and developer-focused content. Please, take a moment to explore the site and share your feedback with us.
  </p>
</div>
{% endblock %}
<!-- ends -->
{% set class = "md-header" %}
{% if "navigation.tabs.sticky" in features %}
  {% set class = class ~ " md-header--lifted" %}
{% endif %}
<header class="{{ class }}" data-md-component="header"  onload="loadVersions()">
    <!--<link rel="stylesheet" href="https://code.kx.com/home/assets/stylesheets/main.b941530a.min.css">-->
    <!--<link rel="stylesheet" href="https://code.kx.com/home/assets/stylesheets/vendor/mermaid.733f213f.min.css">-->
  <nav class="md-header__inner md-grid" aria-label="{{ lang.t('header.title') }}">
    <a href="https://code.kx.com/" title="code.kx.com" class="md-header__button md-logo" aria-label="{{ config.site_name }}" data-md-component="logo">
      {% include "partials/logo.html" %}
    </a>
    <label class="md-header__button md-icon" for="__drawer">
      {% include ".icons/material/menu" ~ ".svg" %}
    </label>
    <div class="md-header__title" data-md-component="header-title">
      <div class="md-header__ellipsis">
        <div class="md-header__topic" style="font-weight:500">
            <span class="md-ellipsis" >
            {{ config.site_name }}
          </span>
          <script>
            var currVersion = '{{ config.curr_docs_version }}'
          </script>
          <style>
            .md-version:focus-within .md-version__list, .md-version:hover .md-version__list {
              max-height: 6rem;
              opacity: 1;
              transition: max-height 0ms,opacity .25s;
            }
          </style>
              <div class="md-version">
                <button class="md-version__current" aria-label="Select version">{{ config.curr_docs_version }}</button>
                <ul class="md-version__list" style="margin-top: 20%;">
                  <div id="versionList"></div>
                </ul>
              </div>
              <div id="latestFlag"></div>
        </div>
        <div class="md-header__topic" data-md-component="header-topic">
          <span class="md-ellipsis">
            {% if page and page.meta and page.meta.title %}
              {{ page.meta.title }}
            {% else %}
              {{ page.title }}
            {% endif %}
          </span>
        </div>
      </div>
    </div>
    {% if not config.theme.palette is mapping %}
      <form class="md-header__option" data-md-component="palette">
        {% for option in config.theme.palette %}
          {% set primary = option.primary | replace(" ", "-") | lower %}
          {% set accent  = option.accent  | replace(" ", "-") | lower %}
          <input class="md-option" data-md-color-media="{{ option.media }}" data-md-color-scheme="{{ option.scheme }}" data-md-color-primary="{{ primary }}" data-md-color-accent="{{ accent }}" {% if option.toggle %} aria-label="{{ option.toggle.name }}" {% else %} aria-hidden="true" {% endif %} type="radio" name="__palette" id="__palette_{{ loop.index }}">
          {% if option.toggle %}
            <label class="md-header__button md-icon" title="{{ option.toggle.name }}" for="__palette_{{ loop.index0 or loop.length }}" hidden>
              {% include ".icons/" ~ option.toggle.icon ~ ".svg" %}
            </label>
          {% endif %}
        {% endfor %}
      </form>
    {% endif %}
    {% if config.extra.alternate %}
      <div class="md-header__option">
        <div class="md-select">
          {% set icon = config.theme.icon.alternate or "material/translate" %}
          <button class="md-header__button md-icon" aria-label="{{ lang.t('select.language.title') }}">
            {% include ".icons/" ~ icon ~ ".svg" %}
          </button>
          <div class="md-select__inner">
            <ul class="md-select__list">
              {% for alt in config.extra.alternate %}
                <li class="md-select__item">
                  <a href="{{ alt.link | url }}" hreflang="{{ alt.lang }}" class="md-select__link">
                    {{ alt.name }}
                  </a>
                </li>
                {% endfor %}
            </ul>
          </div>
        </div>
      </div>
    {% endif %}
    {% if "search" in config["plugins"] %}
      <label class="md-header__button md-icon" for="__search">
        {% include ".icons/material/magnify.svg" %}
      </label>
      {% include "partials/search.html" %}
    {% endif %}
    {% if config.repo_url %}
      <div class="md-header__source">
        {% include "partials/source.html" %}
      </div>
    {% endif %}
  </nav>
  {% if "navigation.tabs.sticky" in features %}
    {% if "navigation.tabs" in features %}
      {% include "partials/tabs.html" %}
    {% endif %}
  {% endif %}
</header>
<style>
    @media (max-width: 430px) {
      .md-typeset span{display:none;}
    }
  /* Button used to open the chat form - fixed at the bottom of the page */
  .open-button {
    background-color: var(--md-typeset-a-color);
    border: none;
    border-radius: 10px;
    bottom: 23px;
    box-sizing: border-box;
    color: var(--md-default-bg-color);
    cursor: pointer;
    font-size: 13px;
    opacity: 0.8;
    padding: 10px 10px;
    position: fixed;
    right: 28px;
    width: 160px;
    z-index: 3;
  }
</style>
<body>
  <button class="open-button md-typeset" onclick="window.location.href='https://community.kx.com/t5/forums/postpage/choose-node/true/interaction-style/forum/override-styles/true/board-id/kxinsights'"><span class="twemoji"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><path d="M256 32C114.6 32 0 125.1 0 240c0 47.6 19.9 91.2 52.9 126.3C38 405.7 7 439.1 6.5 439.5c-6.6 7-8.4 17.2-4.6 26S14.4 480 24 480c61.5 0 110-25.7 139.1-46.3C192 442.8 223.2 448 256 448c141.4 0 256-93.1 256-208S397.4 32 256 32zm0 368c-26.7 0-53.1-4.1-78.4-12.1l-22.7-7.2-19.5 13.8c-14.3 10.1-33.9 21.4-57.5 29 7.3-12.1 14.4-25.7 19.9-40.2l10.6-28.1-20.6-21.8C69.7 314.1 48 282.2 48 240c0-88.2 93.3-160 208-160s208 71.8 208 160-93.3 160-208 160z"></path></svg></span>&nbsp;Ask a question</button>
  </body>
