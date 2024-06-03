"use client";

import Header from "./_component/Header";
import Output from "./_component/Output";

import { ArrowRightCircleIcon, TrashIcon } from "@heroicons/react/24/outline";
import { OutputType } from "./_lib";
import { useState } from "react";
import OutputDetailModal from "./_component/OutputDetailModal";
import { generate } from "./_lib/api";

export default function Home() {
  const [id, setId] = useState(0);
  const [outputs, setOutputs] = useState<OutputType[]>([]);
  const [context, setContext] = useState(
    'THE flax was in full bloom. It had pretty little blue flowers, as delicate as the wings of a moth. The sun shone on it and the showers watered it. This was as good for the flax as it is for little children to be washed and then kissed by their mothers. They look much prettier for it, and so did the flax. "People say that I look exceedingly well," said the flax, "and that I am so fine and long that I shall make a beautiful piece of linen. How fortunate I am! It makes me so happy to know that something can be made of me. How the sunshine cheers me, and how sweet and refreshing is the rain! My happiness overpowers me! No one in the world can feel happier than I."'
  );
  const [open, setOpen] = useState<number | null>(null);

  const onSubmit = async () => {
    const qnas = (await generate(context)).qnas;
    setOutputs((prev) => [
      ...prev,
      {
        id: id,
        question: qnas[0].question,
        answer: qnas[0].answer,
        like: null,
      },
      {
        id: id + 1,
        question: qnas[1].question,
        answer: qnas[1].answer,
        like: null,
      },
      {
        id: id + 2,
        question: qnas[2].question,
        answer: qnas[2].answer,
        like: null,
      },
    ]);
    setId((prev) => prev + 3);
  };

  const onClear = () => {
    if (!confirm("Are you sure you want to clear all questions?")) return;
    setOutputs([]);
  };

  const onChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setContext(event.target.value);
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
          value={context}
          onChange={onChange}
        />
      </section>
      <section className="flex-1 max-w-4c" />
      <main className="flex-[3] max-w-8c mx-auto py-16 flex flex-col gap-16">
        <Header />
        {open !== null && (
          <OutputDetailModal
            output={outputs.find((output) => output.id === open)!}
            setOpen={setOpen}
          />
        )}
        <section className="flex flex-col gap-8">
          {outputs.map((output) => (
            <Output
              key={output.id}
              output={output}
              setOutputs={setOutputs}
              setOpen={setOpen}
            />
          ))}
        </section>
      </main>
    </>
  );
}
