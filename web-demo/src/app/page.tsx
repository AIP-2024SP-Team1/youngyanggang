import Header from "./_component/Header";
import Output from "./_component/Output";

import { ArrowPathIcon } from "@heroicons/react/24/solid";

const output_sample = {
  id: 1,
  question: "What is the flax in full bloom?",
  answer:
    "The flax in full bloom is a plant with pretty little blue flowers, described as delicate as the wings of a moth. It is thriving under the sun and showers, which make it look even more beautiful and contribute to its growth, making it fine and long, ideal for producing a beautiful piece of linen. The flax is expressing its happiness and fortune due to its flourishing condition and the potential it holds for becoming something valuable.",
};

export default function Home() {
  return (
    <>
      <section className="flex-1 max-w-4c p-8 flex flex-col gap-16 bg-white">
        <div className="flex items-center justify-between">
          <h2 className="font-bold">context</h2>
          <ArrowPathIcon className="w-4 h-4 cursor-pointer" />
        </div>
        <textarea
          className="bg-gray-50 focus:outline-none"
          name="context"
          id="context"
          rows={20}
          value={`THE flax was in full bloom. It had pretty little blue flowers, as delicate as the wings of a moth. The sun shone on it and the showers watered it. This was as good for the flax as it is for little children to be washed and then kissed by their mothers. They look much prettier for it, and so did the flax. "People say that I look exceedingly well," said the flax, "and that I am so fine and long that I shall make a beautiful piece of linen. How fortunate I am! It makes me so happy to know that something can be made of me. How the sunshine cheers me, and how sweet and refreshing is the rain! My happiness overpowers me! No one in the world can feel happier than I.`}
        ></textarea>
      </section>
      <div className="flex-[3] max-w-8c mx-auto flex flex-col gap-16">
        <Header />
        <main className="flex flex-col gap-8">
          <Output output={output_sample} />
        </main>
      </div>
    </>
  );
}
