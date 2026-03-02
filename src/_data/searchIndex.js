module.exports = function (data) {
  const posts = (data.collections?.posts || []).slice().sort((a, b) => b.date - a.date);
  return posts.map((post) => ({
    title: post.data.title,
    description: post.data.description || "",
    url: post.url,
    date: post.date ? post.date.toISOString().split("T")[0] : "",
    category: post.data.category || "",
    tags: Array.isArray(post.data.tags) ? post.data.tags.filter((tag) => tag !== "posts") : "",
    content: post.templateContent ? post.templateContent.replace(/<[^>]*>/g, " ") : ""
  }));
};
