import { auth } from "@/app/(auth)/auth";
import { getChunksByFilePaths } from "@/app/db";
import { openai } from "@ai-sdk/openai";
import {
  cosineSimilarity,
  embed,
  Experimental_LanguageModelV1Middleware,
  generateObject,
  generateText,
} from "ai";
import { z } from "zod";

// schema for validating the custom provider metadata
const selectionSchema = z.object({
  files: z.object({
    selection: z.array(z.string()),
  }),
});

export const ragMiddleware: Experimental_LanguageModelV1Middleware = {
  transformParams: async ({ params }) => {
    const session = await auth();

    if (!session) return params; // no user session

    const { prompt: messages, providerMetadata } = params;

    console.log("Provider Metadata", providerMetadata);

    // validate the provider metadata with Zod:
    const { success, data } = selectionSchema.safeParse(providerMetadata);

    if (!success) return params; // no files selected

    const selection = data.files.selection;

    console.log("Selection", selection);

    const recentMessage = messages.pop();

    // Make sure the last message is a user message before proceeding
    if (!recentMessage || recentMessage.role !== "user") {
      if (recentMessage) {
        messages.push(recentMessage);
      }

      return params;
    }

    // Get the content of the last user message
    const lastUserMessageContent = recentMessage.content
      .filter((content) => content.type === "text")
      .map((content) => content.text)
      .join("\n");

    // Classify the user prompt as whether it requires more context or not
    const { object: classification } = await generateObject({
      // fast model for classification:
      model: openai("gpt-4o-mini", { structuredOutputs: true }),
      output: "enum",
      enum: ["question", "statement", "other"],
      system: "classify the user message as a question, statement, or other",
      prompt: lastUserMessageContent,
    });

    //console.log it was a question if it was a question
    if (classification === "question") {
      console.log("It was a question");
    }
    //console.log it was a statement if it was a statement
    if (classification === "statement") {
      console.log("It was a statement");
    }
    //console.log it was other if it was other
    if (classification === "other") {
      console.log("It was other");
    }

    // only use RAG for questions
    if (classification !== "question") {
      messages.push(recentMessage);
      return params;
    }

    // Use hypothetical document embeddings:
    const { text: hypotheticalAnswer } = await generateText({
      // fast model for generating hypothetical answer:
      model: openai("gpt-4o-mini", { structuredOutputs: true }),
      system: `Answer the users question:
          - if you need more information, say "i need more"
          - if you're sorry, say "i'm sorry"
          - if you can answer the question, answer it
      `,
      prompt: lastUserMessageContent,
    });

    console.log("Hypothetical Answer", hypotheticalAnswer);

    // if the hypotheticalAnswer contains "i'm sorry" or "i need more" return the hypotheticalAnswer
    if (hypotheticalAnswer.includes("i'm sorry") || hypotheticalAnswer.includes("i need more")) {
      messages.push(recentMessage);
      return params;
    }


    // Embed the hypothetical answer
    const { embedding: hypotheticalAnswerEmbedding } = await embed({
      model: openai.embedding("text-embedding-3-small"),
      value: hypotheticalAnswer,
    });

    console.log("Hypothetical Answer embedded");

    // find relevant chunks based on the selection
    const chunksBySelection = await getChunksByFilePaths({
      filePaths: selection.map((path) => `${session.user?.email}/${path}`),
    });

    console.log("Chunks by selection", chunksBySelection.length);

    const chunksWithSimilarity = chunksBySelection.map((chunk) => ({
      ...chunk,
      similarity: cosineSimilarity(
        hypotheticalAnswerEmbedding,
        chunk.embedding,
      ),
    }));

    console.log("Chunks with similarity", chunksWithSimilarity.length);

    // rank the chunks by similarity and take the top K
    chunksWithSimilarity.sort((a, b) => b.similarity - a.similarity);
    const k = 10;
    const topKChunks = chunksWithSimilarity.slice(0.5, k);

    console.log("Top K Chunks", topKChunks.map((chunk) => ({
      similarity: chunk.similarity,
      filePath: chunk.filePath,
    })));

    // add the chunks to the last user message
    messages.push({
      role: "user",
      content: [
        ...recentMessage.content,
        {
          type: "text",
          text: "Here is some relevant information that you can use to answer the question:",
        },
        ...topKChunks.map((chunk) => ({
          type: "text" as const,
          text: chunk.content,
        })),
      ],
    });

    console.log("Messages", messages);
    console.log("Params", params);

    return { ...params, prompt: messages };
  },
};
