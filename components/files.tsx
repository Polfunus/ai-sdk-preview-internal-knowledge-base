"use client";

import useSWR from "swr";
import {
  CheckedSquare,
  LoaderIcon,
  TrashIcon,
  UncheckedSquare,
  UploadIcon,
} from "./icons";
import { Dispatch, SetStateAction, useRef, useState } from "react";
import { fetcher } from "@/utils/functions";
import cx from "classnames";
import { motion } from "framer-motion";

export const Files = ({
  selectedFilePathnames,
  setSelectedFilePathnames,
}: {
  selectedFilePathnames: string[];
  setSelectedFilePathnames: Dispatch<SetStateAction<string[]>>;
  setIsDropdownVisible: Dispatch<SetStateAction<boolean>>;
}) => {
  const inputFileRef = useRef<HTMLInputElement>(null);
  const [uploadQueue, setUploadQueue] = useState<Array<string>>([]);
  const { data: files, mutate } = useSWR("api/files/list", fetcher, {
    fallbackData: [],
  });

  return (
    <motion.div
      className="fixed top-0 right-0 w-dvw h-dvh p-4 flex flex-col gap-6 bg-white dark:bg-zinc-800  z-30"
      initial={{ y: "100%" }}
      animate={{ y: "50%" }}
      exit={{ y: "100%" }}
      transition={{ type: "spring", stiffness: 400, damping: 40 }}
    >
      <div className="flex flex-row justify-between items-center">
        <div className="text-sm flex flex-row gap-2">
          <div className="text-zinc-900 dark:text-zinc-300">Uploads</div>
          <div className="text-zinc-500 dark:text-zinc-400">
            {files.length} Files
          </div>
        </div>

        <input
          name="file"
          ref={inputFileRef}
          type="file"
          required
          className="opacity-0 pointer-events-none w-1"
          multiple={false}
          onChange={async (event) => {
            const file = event.target.files![0];

            if (file) {
              setUploadQueue((currentQueue) => [...currentQueue, file.name]);

              await fetch(`/api/files/upload?filename=${file.name}`, {
                method: "POST",
                body: file,
              });

              setUploadQueue((currentQueue) =>
                currentQueue.filter((filename) => filename !== file.name),
              );

              mutate();
            }
          }}
        />

        <div
          className="bg-zinc-900 text-zinc-50 hover:bg-zinc-800 flex flex-row gap-2 items-center dark:text-zinc-800 text-sm dark:bg-zinc-100 rounded-md p-1 px-2 dark:hover:bg-zinc-200 cursor-pointer"
          onClick={() => {
            inputFileRef.current?.click();
          }}
        >
          <UploadIcon />
          <div>Upload a file</div>
        </div>
      </div>

      <div className="flex flex-col">
        {files.map((file: any) => (
          <div
            key={file.pathname}
            className={`flex flex-row gap-4 items-center p-2 border-b dark:border-zinc-700 ${
              selectedFilePathnames.includes(file.pathname)
                ? "bg-zinc-100 dark:bg-zinc-700"
                : ""
            }`}
          >
            <div
              className={cx(
                "cursor-pointer",
                selectedFilePathnames.includes(file.pathname)
                  ? "text-blue-600 dark:text-zinc-50"
                  : "text-zinc-500",
              )}
              onClick={() => {
                setSelectedFilePathnames((currentSelections) => {
                  if (currentSelections.includes(file.pathname)) {
                    return currentSelections.filter(
                      (path) => path !== file.pathname,
                    );
                  } else {
                    return [...currentSelections, file.pathname];
                  }
                });
              }}
            >
              {selectedFilePathnames.includes(file.pathname) ? (
                <CheckedSquare />
              ) : (
                <UncheckedSquare />
              )}
            </div>

            <div className="flex flex-row justify-between w-full">
              <div className="text-sm text-zinc-500 dark:text-zinc-400">
                {file.pathname}
              </div>
            </div>

            <div
              className="text-zinc-500 hover:bg-zinc-100 dark:text-zinc-500 hover:dark:bg-zinc-700 hover:text-red-500 p-1 cursor-pointer rounded-md"
              onClick={async () => {
                await fetch(`/api/files/delete?fileurl=${file.url}`, {
                  method: "DELETE",
                });

                mutate();
              }}
            >
              <TrashIcon />
            </div>
          </div>
        ))}

        {uploadQueue.map((fileName) => (
          <div
            key={fileName}
            className="flex flex-row justify-between p-1.5 px-2 gap-4 items-center"
          >
            <div className="text-zinc-500">
              <div className="animate-spin">
                <LoaderIcon />
              </div>
            </div>

            <div className="flex flex-row justify-between w-full">
              <div className="text-sm text-zinc-400 dark:text-zinc-400">
                {fileName}
              </div>
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
};