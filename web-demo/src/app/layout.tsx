import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Multimodal7b Demo",
  description: "Generate questions with multimodal7b",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="flex bg-gray-50 min-h-dvh text-gray-950">
        {children}
      </body>
    </html>
  );
}
