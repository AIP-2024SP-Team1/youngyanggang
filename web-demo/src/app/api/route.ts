import csv from "csv-parser";
import fs from "fs";
import { NextResponse } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

interface Row {
  id: number;
  context: string;
  questions: string[];
}

async function loadData(): Promise<Row[]> {
  const results: Row[] = [];

  const stream = fs
    .createReadStream("../multimodal7b/output/result(v2).csv")
    .pipe(csv())
    .on("data", (result: any) =>
      results.push({
        id: parseInt(result[""]),
        context: result.context,
        questions: result.generated.split(", "),
      })
    );

  return new Promise((resolve) => {
    stream.on("end", function () {
      resolve(results);
    });
  });
}

export async function GET(request: Request) {
  const data = await loadData();
  return NextResponse.json(data, { status: 200 });
}

export async function POST(request: Request) {
  const context: string = (await request.json()).context;
  const data = await loadData();
  const row = data.find((row) => row.context === context);

  const completion = await openai.chat.completions.create({
    messages: [
      {
        role: "system",
        content: "You are a helpful assistant designed to output JSON.",
      },
      {
        role: "user",
        content: `For each question, generate appropriate answer paired considering the context.\ncontext: ${context}\n${row?.questions.map(
          (question, idx) => `question ${idx}: ${question}`
        )}`,
      },
    ],
    model: "gpt-3.5-turbo-0125",
    response_format: { type: "json_object" },
  });

  const content = await JSON.parse(completion.choices[0].message.content!);

  let qnas = content.answers;
  if (!content.answers[0].question)
    qnas = Object.values(content.answers).map((answer, idx) => ({
      question: row?.questions[idx],
      answer: answer,
    }));

  console.log(qnas);
  return NextResponse.json({ qnas }, { status: 200 });
}
