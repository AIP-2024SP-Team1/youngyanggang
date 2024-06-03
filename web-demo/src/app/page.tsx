"use client";

import Header from "./_component/Header";
import Output from "./_component/Output";

import { ArrowRightCircleIcon, TrashIcon } from "@heroicons/react/24/outline";
import { OutputType } from "./_lib";
import { useState } from "react";

const outputs_sample: OutputType[] = [
  {
    id: 1,
    question: "What is the flax in full bloom?",
    answer:
      "The flax in full bloom is a plant with pretty little blue flowers, described as delicate as the wings of a moth. It is thriving under the sun and showers, which make it look even more beautiful and contribute to its growth, making it fine and long, ideal for producing a beautiful piece of linen. The flax is expressing its happiness and fortune due to its flourishing condition and the potential it holds for becoming something valuable.",
    like: null,
  },
  {
    id: 2,
    question: "What is the flax in full bloom?",
    answer:
      "The flax in full bloom is a plant with pretty little blue flowers, described as delicate as the wings of a moth. It is thriving under the sun and showers, which make it look even more beautiful and contribute to its growth, making it fine and long, ideal for producing a beautiful piece of linen. The flax is expressing its happiness and fortune due to its flourishing condition and the potential it holds for becoming something valuable.",
    like: null,
  },
  {
    id: 3,
    question: "What is the flax in full bloom?",
    answer:
      "The flax in full bloom is a plant with pretty little blue flowers, described as delicate as the wings of a moth. It is thriving under the sun and showers, which make it look even more beautiful and contribute to its growth, making it fine and long, ideal for producing a beautiful piece of linen. The flax is expressing its happiness and fortune due to its flourishing condition and the potential it holds for becoming something valuable.",
    like: null,
  },
  {
    id: 4,
    question: "What is the flax in full bloom?",
    answer:
      "The flax in full bloom is a plant with pretty little blue flowers, described as delicate as the wings of a moth. It is thriving under the sun and showers, which make it look even more beautiful and contribute to its growth, making it fine and long, ideal for producing a beautiful piece of linen. The flax is expressing its happiness and fortune due to its flourishing condition and the potential it holds for becoming something valuable.",
    like: null,
  },
  {
    id: 5,
    question: "What is the flax in full bloom?",
    answer:
      "The flax in full bloom is a plant with pretty little blue flowers, described as delicate as the wings of a moth. It is thriving under the sun and showers, which make it look even more beautiful and contribute to its growth, making it fine and long, ideal for producing a beautiful piece of linen. The flax is expressing its happiness and fortune due to its flourishing condition and the potential it holds for becoming something valuable.",
    like: null,
  },
  {
    id: 6,
    question: "What is the flax in full bloom?",
    answer:
      "The flax in full bloom is a plant with pretty little blue flowers, described as delicate as the wings of a moth. It is thriving under the sun and showers, which make it look even more beautiful and contribute to its growth, making it fine and long, ideal for producing a beautiful piece of linen. The flax is expressing its happiness and fortune due to its flourishing condition and the potential it holds for becoming something valuable.",
    like: null,
  },
  {
    id: 7,
    question: "What is the flax in full bloom?",
    answer:
      "The flax in full bloom is a plant with pretty little blue flowers, described as delicate as the wings of a moth. It is thriving under the sun and showers, which make it look even more beautiful and contribute to its growth, making it fine and long, ideal for producing a beautiful piece of linen. The flax is expressing its happiness and fortune due to its flourishing condition and the potential it holds for becoming something valuable.",
    like: null,
  },
];

export default function Home() {
  const [id, setId] = useState(8); // id starts from 8
  const [outputs, setOutputs] = useState<OutputType[]>(outputs_sample);

  const onSubmit = () => {
    setOutputs((prev) => [
      ...prev,
      {
        id,
        question: "What is the flax in full bloom?",
        answer:
          "The flax in full bloom is a plant with pretty little blue flowers, described as delicate as the wings of a moth. It is thriving under the sun and showers, which make it look even more beautiful and contribute to its growth, making it fine and long, ideal for producing a beautiful piece of linen. The flax is expressing its happiness and fortune due to its flourishing condition and the potential it holds for becoming something valuable.",
        like: null,
      },
    ]);
    setId((prev) => prev + 1);
  };

  const onClear = () => {
    if (!confirm("Are you sure you want to clear all questions?")) return;
    setOutputs([]);
  };

  return (
    <>
      <section className="fixed min-w-2c w-[25%] max-w-4c h-dvh p-8 flex flex-col gap-16 bg-white">
        <div className="flex items-center justify-between">
          <h2 className="font-bold">context</h2>
          <div className="flex items-center gap-4">
            <ArrowRightCircleIcon
              className="w-5 h-5 cursor-pointer"
              onClick={onSubmit}
            />
            <TrashIcon className="w-4 h-4 cursor-pointer" onClick={onClear} />
          </div>
        </div>
        <textarea
          className="bg-gray-50 focus:outline-none"
          name="context"
          id="context"
          rows={20}
          value={`THE flax was in full bloom. It had pretty little blue flowers, as delicate as the wings of a moth. The sun shone on it and the showers watered it. This was as good for the flax as it is for little children to be washed and then kissed by their mothers. They look much prettier for it, and so did the flax. "People say that I look exceedingly well," said the flax, "and that I am so fine and long that I shall make a beautiful piece of linen. How fortunate I am! It makes me so happy to know that something can be made of me. How the sunshine cheers me, and how sweet and refreshing is the rain! My happiness overpowers me! No one in the world can feel happier than I.`}
        />
      </section>
      <section className="flex-1 max-w-4c" />
      <main className="flex-[3] max-w-8c mx-auto py-16 flex flex-col gap-16">
        <Header />
        <section className="flex flex-col gap-8">
          {outputs.map((output) => (
            <Output output={output} setOutputs={setOutputs} />
          ))}
        </section>
      </main>
    </>
  );
}
