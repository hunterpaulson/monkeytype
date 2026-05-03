export type EnvConfig = {
  backendUrl: string;
  isDevelopment: boolean;
  isTypeGptDemo: boolean;
  clientVersion: string;
  recaptchaSiteKey: string;
  typeGptWeightsBaseUrl: string | undefined;
  quickLoginEmail: string | undefined;
  quickLoginPassword: string | undefined;
};

declare module "virtual:env-config" {
  export const envConfig: EnvConfig;
}
