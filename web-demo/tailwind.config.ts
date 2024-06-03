import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        /* Add new colors */
      },
      fontFamily: {
        pretendard: ["pretendard", "gothic"],
      },
      spacing: {
        "1c": "4.875rem",
        "2c": "11.25rem",
        "3c": "17.625rem",
        "4c": "24rem",
        "6c": "36.75rem",
        "8c": "49.5rem",
        "10c": "62.25rem",
        "12c": "75rem",
      },
    },
  },
  plugins: [],
};
export default config;
