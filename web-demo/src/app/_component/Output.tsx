import { OutputType } from "../_lib";

import {
  HandThumbDownIcon,
  HandThumbUpIcon,
  ClipboardIcon,
  MinusCircleIcon,
} from "@heroicons/react/24/outline";

interface Props {
  output: OutputType;
}

export default function Output({ output }: Props) {
  return (
    <div className="flex flex-col">
      <div className="flex items-center justify-between p-2">
        <h3>Question {output.id}</h3>
        <menu className="flex gap-4">
          <HandThumbDownIcon className="w-4 h-4 cursor-pointer hover:scale-105" />
          <HandThumbUpIcon className="w-4 h-4 cursor-pointer hover:scale-105" />
          <ClipboardIcon className="w-4 h-4 cursor-pointer hover:scale-105" />
          <MinusCircleIcon className="w-4 h-4 cursor-pointer hover:scale-105" />
        </menu>
      </div>
      <div className="bg-white rounded-lg p-4 cursor-pointer hover:-translate-y-[2px] transition-all">
        {output.question}
      </div>
    </div>
  );
}
