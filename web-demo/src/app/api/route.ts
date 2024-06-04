import { randomInt } from "crypto";
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
  const idx = randomInt(0, data.length - 1);
  return NextResponse.json(data[idx].context, { status: 200 });
}

export async function POST(request: Request) {
  const context: string = (await request.json()).context;
  const data = await loadData();
  const row = data.find((row) => row.context === context);
  console.log(row?.questions);

  const input = `For each question, generate appropriate answer paired considering the context.
### Context:
${context}
### Questions:
${row?.questions.map((question, idx) => `question ${idx}: ${question}\n`)}
### Response
qnas: { question: string; answer: string; }[];`;

  const completion = await openai.chat.completions.create({
    messages: [
      {
        role: "system",
        content: "You are a helpful assistant designed to output JSON.",
      },
      {
        role: "user",
        content: input,
      },
    ],
    model: "gpt-3.5-turbo-0125",
    response_format: { type: "json_object" },
  });

  const generated = await JSON.parse(completion.choices[0].message.content!);
  console.log(generated);

  const qnas = Object.values(generated.qnas);

  return NextResponse.json({ qnas }, { status: 200 });
}
