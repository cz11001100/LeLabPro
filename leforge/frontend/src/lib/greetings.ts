// Rotating home-page greeting. Templates that carry `{name}` address the user
// by their HuggingFace handle; the rest read fine on their own. Only some
// lines carry an emoji so the headline does not turn into a sticker sheet.
export const GREETINGS: string[] = [
  "Good morning, {name} ☀️",
  "Rise and shine, {name}",
  "Welcome back, {name} 👋",
  "The sun is new each day",
  "Every morning, a new page. Write a good one, {name}",
  "Make today count",
  "Someday is not a day of the week",
  "The best way out is always through",
  "Ordinary, yet unique. Hello, {name}",
  "Even in darkness, make light ✨",
  "Don't wait for opportunity. Build it",
  "Slow is fine, as long as you don't stop",
  "Begin again, {name} 🔄",
  "Hello, {name} — let's teach another robot 🤖",
  "Coffee first, gradients later, {name} ☕",
  "The breeze at dawn has secrets",
  "Yesterday is gone. New weights today",
  "Progress, not perfection",
  "Good things are loading, {name} ⏳",
  "Another day to bend the world",
  "It always seems impossible until it's done",
  "Well begun is half done, {name}",
  "Fail fast, learn faster",
  "One more episode, {name}",
  "The best time to start was yesterday; next best is now",
  "Keep your sunny side up 🌻",
  "Every sunrise is a retry",
  "Practice doesn't make perfect, it makes permanent",
  "Slow is smooth, smooth is fast",
  "Trust the data, then check it again",
  "Let the robot do the boring half 🤖",
  "What you do today compounds",
  "Start where you are, use what you have",
  "Do the hard thing first, {name}",
  "A year from now you'll wish you started today",
  "Nothing splendid is achieved without effort",
  "Small steps, tight loops 🔁",
  "Done is better than perfect",
  "Half coffee, full focus ☕",
  "Ship the ugly version today",
];

// Avoid handing back the same line twice in a row when the user navigates home.
let lastIndex = -1;

export const randomGreeting = (name: string): string => {
  let index = Math.floor(Math.random() * GREETINGS.length);
  if (GREETINGS.length > 1 && index === lastIndex) {
    index = (index + 1) % GREETINGS.length;
  }
  lastIndex = index;
  const template = GREETINGS[index];
  const line = name
    ? template.split("{name}").join(name)
    : template.split(", {name}").join("");
  return `"${line}"`;
};
