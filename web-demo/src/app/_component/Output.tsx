"use client";

import { OutputType } from "../_lib";
import {
  HandThumbDownIcon,
  HandThumbUpIcon,
  ClipboardIcon,
  MinusCircleIcon,
} from "@heroicons/react/24/outline";

interface Props {
  output: OutputType;
  setOutputs: React.Dispatch<React.SetStateAction<OutputType[]>>;
}

export default function Output({ output, setOutputs }: Props) {
  const onUpvote = () => {
    setOutputs((prev) => {
      const index = prev.findIndex((item) => item.id === output.id);
      prev[index].upvotes += 1;
      return [...prev];
    });

    alert("Upvoted!");
  };
  const onDownvote = () => {
    setOutputs((prev) => {
      const index = prev.findIndex((item) => item.id === output.id);
      prev[index].downvotes += 1;
      return [...prev];
    });

    alert("Downvoted!");
  };

  const onCopy = () => {
    navigator.clipboard.writeText(output.question);

    alert("Copied to clipboard!");
  };

  const onRemove = () => {
    setOutputs((prev) => prev.filter((item) => item.id !== output.id));
  };

  return (
    <div className="flex flex-col">
      <div className="flex items-center justify-between p-2">
        <h3>Question {output.id}</h3>
        <menu className="flex gap-4">
          <HandThumbUpIcon
            className="w-4 h-4 cursor-pointer hover:scale-105"
            onClick={onUpvote}
          />
          <HandThumbDownIcon
            className="w-4 h-4 cursor-pointer hover:scale-105"
            onClick={onDownvote}
          />
          <ClipboardIcon
            className="w-4 h-4 cursor-pointer hover:scale-105"
            onClick={onCopy}
          />
          <MinusCircleIcon
            className="w-4 h-4 cursor-pointer hover:scale-105"
            onClick={onRemove}
          />
        </menu>
      </div>
      <div className="bg-white rounded-lg p-4 cursor-pointer hover:-translate-y-[2px] transition-all">
        {output.question}
      </div>
    </div>
  );
}
