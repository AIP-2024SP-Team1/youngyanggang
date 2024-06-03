"use client";

import { OutputType } from "../_lib";
import { ClipboardIcon, XMarkIcon } from "@heroicons/react/24/outline";

interface Props {
  output: OutputType;
  setOpen: React.Dispatch<React.SetStateAction<number | null>>;
}

export default function OutputDetailModal({ output, setOpen }: Props) {
  const onClose = () => {
    setOpen(null);
  };

  const onCopy = () => {
    navigator.clipboard.writeText(output.answer);
    alert("Copied answer to clipboard!");
  };

  return (
    <>
      <div
        className="fixed top-0 left-0 w-dvw h-dvh bg-black opacity-50 z-10"
        onClick={onClose}
      />
      <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8c flex flex-col gap-4 bg-white rounded-lg p-4 z-20">
        <div className="flex justify-between">
          <h3 className="font-bold">Question:</h3>
          <XMarkIcon
            className="w-5 h-5 cursor-pointer self-end"
            onClick={onClose}
          />
        </div>
        <p>{output.question}</p>
        <hr />
        <div className="flex justify-between">
          <h3 className="font-bold">Answer:</h3>
          <ClipboardIcon
            className="w-5 h-5 cursor-pointer self-end"
            onClick={onCopy}
          />
        </div>
        <p>{output.answer}</p>
      </div>
    </>
  );
}
