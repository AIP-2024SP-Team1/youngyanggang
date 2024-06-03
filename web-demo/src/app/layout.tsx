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
      <body className="flex min-h-[100dvh] w-full max-w-12c m-auto flex-col gap-12">
        {children}
      </body>
    </html>
  );
}
