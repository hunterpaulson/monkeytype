import { envConfig } from "virtual:env-config";

export function isDevEnvironment(): boolean {
  return envConfig.isDevelopment;
}

export function isTypeGptDemo(): boolean {
  return envConfig.isTypeGptDemo;
}
