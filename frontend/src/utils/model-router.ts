// model-router.ts

/**
 * Model routing logic based on task type with OpenRouter enforced.
 */

interface ModelRoute {
  model: string;
  apiBaseUrl: string;
  maxInputTokens: number;
  maxOutputTokens: number;
  reason: string;
}

export type TaskType =
  | "code-refactor"
  | "type-gen"
  | "controller-synthesis"
  | "context-extract"
  | "summary"
  | "rag-query"
  | "unit-test"
  | "lint"
  | "boilerplate"
  | "fallback";

const OPENROUTER_BASE = "https://openrouter.ai/api/v1";

export function getModelForTask(task: TaskType): ModelRoute {
  switch (task) {
    case "code-refactor":
    case "controller-synthesis":
    case "context-extract":
      return {
        model: "openai/gpt-4o",
        apiBaseUrl: OPENROUTER_BASE,
        maxInputTokens: 16000,
        maxOutputTokens: 2000,
        reason: "High reasoning refactor via GPT-4o through OpenRouter"
      };
    case "summary":
    case "rag-query":
      return {
        model: "anthropic/claude-3-opus",
        apiBaseUrl: OPENROUTER_BASE,
        maxInputTokens: 100000,
        maxOutputTokens: 3000,
        reason: "Long-context summary via Claude-3 through OpenRouter"
      };
    case "type-gen":
    case "unit-test":
    case "lint":
      return {
        model: "openai/gpt-3.5-turbo",
        apiBaseUrl: OPENROUTER_BASE,
        maxInputTokens: 8000,
        maxOutputTokens: 1000,
        reason: "Lightweight utility generation via GPT-3.5 through OpenRouter"
      };
    case "boilerplate":
    case "fallback":
      return {
        model: "mistral/mixtral-8x7b",
        apiBaseUrl: OPENROUTER_BASE,
        maxInputTokens: 4000,
        maxOutputTokens: 800,
        reason: "Default fallback for budget-safe logic"
      };
    default:
      return {
        model: "openai/gpt-4o-mini",
        apiBaseUrl: OPENROUTER_BASE,
        maxInputTokens: 4000,
        maxOutputTokens: 800,
        reason: "Safe default route via GPT-4o-mini on OpenRouter"
      };
  }
}
