module.exports = function (eleventyConfig) {
  eleventyConfig.addFilter("readableDate", (dateObj) => {
    return new Intl.DateTimeFormat("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit"
    }).format(dateObj);
  });

  eleventyConfig.addFilter("dateISO", (dateObj) => {
    return dateObj.toISOString().split("T")[0];
  });

  eleventyConfig.addFilter("slug", (value) => {
    return String(value || "")
      .trim()
      .toLowerCase()
      .replace(/\s+/g, "-")
      .replace(/[^\w\u4e00-\u9fa5-]/g, "");
  });

  eleventyConfig.addFilter("toJson", (value) => {
    return JSON.stringify(value);
  });

  eleventyConfig.addPassthroughCopy({ "src/assets": "assets" });

  eleventyConfig.addShortcode("bilibili", (bvid) => {
    if (!bvid) return "";
    const src = `https://player.bilibili.com/player.html?bvid=${encodeURIComponent(bvid)}&page=1&high_quality=1`;
    return `<div class="embed"><iframe src="${src}" scrolling="no" frameborder="0" allowfullscreen="true"></iframe></div>`;
  });

  eleventyConfig.addCollection("postsByYear", (collectionApi) => {
    const posts = collectionApi.getFilteredByTag("posts").slice().sort((a, b) => b.date - a.date);
    const groups = new Map();
    for (const post of posts) {
      const year = String(post.date.getFullYear());
      if (!groups.has(year)) groups.set(year, []);
      groups.get(year).push(post);
    }
    return Array.from(groups.entries()).map(([year, items]) => ({
      year,
      posts: items
    }));
  });

  eleventyConfig.addCollection("categoryList", (collectionApi) => {
    const posts = collectionApi.getFilteredByTag("posts").slice().sort((a, b) => b.date - a.date);
    const groups = new Map();
    for (const post of posts) {
      const categories = Array.isArray(post.data.category)
        ? post.data.category
        : post.data.category
        ? [post.data.category]
        : Array.isArray(post.data.categories)
        ? post.data.categories
        : post.data.categories
        ? [post.data.categories]
        : [];
      for (const category of categories) {
        if (!category) continue;
        if (!groups.has(category)) groups.set(category, []);
        groups.get(category).push(post);
      }
    }
    return Array.from(groups.entries())
      .map(([name, items]) => ({ name, posts: items }))
      .sort((a, b) => a.name.localeCompare(b.name, "zh-CN"));
  });

  eleventyConfig.addCollection("tagList", (collectionApi) => {
    const posts = collectionApi.getFilteredByTag("posts").slice().sort((a, b) => b.date - a.date);
    const groups = new Map();
    for (const post of posts) {
      const tags = Array.isArray(post.data.tags) ? post.data.tags : [];
      for (const tag of tags) {
        if (!tag || tag === "posts") continue;
        if (!groups.has(tag)) groups.set(tag, []);
        groups.get(tag).push(post);
      }
    }
    return Array.from(groups.entries())
      .map(([name, items]) => ({ name, posts: items }))
      .sort((a, b) => a.name.localeCompare(b.name, "zh-CN"));
  });

  const sortByTitle = (items) =>
    items.slice().sort((a, b) => String(a.data.title || "").localeCompare(String(b.data.title || ""), "zh-CN"));

  eleventyConfig.addCollection("learningList", (collectionApi) => {
    return sortByTitle(collectionApi.getFilteredByGlob("src/learning/*.md"));
  });

  eleventyConfig.addCollection("projectsList", (collectionApi) => {
    return sortByTitle(collectionApi.getFilteredByGlob("src/projects/*.md"));
  });

  eleventyConfig.addCollection("explorationsList", (collectionApi) => {
    return sortByTitle(collectionApi.getFilteredByGlob("src/explorations/*.md"));
  });

  return {
    dir: {
      input: "src",
      includes: "_includes",
      data: "_data",
      output: "dist"
    },
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
    dataTemplateEngine: "njk"
  };
};
