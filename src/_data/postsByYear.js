module.exports = function (data) {
  const posts = (data.collections?.posts || []).slice().sort((a, b) => b.date - a.date);
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
};
