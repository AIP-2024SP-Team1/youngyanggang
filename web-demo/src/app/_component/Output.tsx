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
  const onLike = () => {
    setOutputs((prev) => {
      const index = prev.findIndex((item) => item.id === output.id);
      prev[index].like = true;
      return [...prev];
    });

    alert("Liked!");
  };
  const onDislike = () => {
    setOutputs((prev) => {
      const index = prev.findIndex((item) => item.id === output.id);
      prev[index].like = false;
      return [...prev];
    });

    alert("Disliked!");
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
            onClick={onLike}
          />
          <HandThumbDownIcon
            className="w-4 h-4 cursor-pointer hover:scale-105"
            onClick={onDislike}
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
