import { Configuration } from "@monkeytype/schemas/configuration";

export const TYPEGPT_SERVER_CONFIGURATION: Configuration = {
  maintenance: false,
  dev: {
    responseSlowdownMs: 0,
  },
  results: {
    savingEnabled: false,
    objectHashCheckEnabled: false,
    filterPresets: {
      enabled: false,
      maxPresetsPerUser: 0,
    },
    limits: {
      regularUser: 1000,
      premiumUser: 10000,
    },
    maxBatchSize: 1000,
  },
  quotes: {
    reporting: {
      enabled: false,
      maxReports: 0,
      contentReportLimit: 0,
    },
    submissionsEnabled: false,
    maxFavorites: 0,
  },
  admin: {
    endpointsEnabled: false,
  },
  apeKeys: {
    endpointsEnabled: false,
    acceptKeys: false,
    maxKeysPerUser: 0,
    apeKeyBytes: 24,
    apeKeySaltRounds: 5,
  },
  users: {
    signUp: false,
    lastHashesCheck: {
      enabled: false,
      maxHashes: 0,
    },
    discordIntegration: {
      enabled: false,
    },
    autoBan: {
      enabled: false,
      maxCount: 5,
      maxHours: 1,
    },
    profiles: {
      enabled: false,
    },
    xp: {
      enabled: false,
      funboxBonus: 0,
      gainMultiplier: 0,
      maxDailyBonus: 0,
      minDailyBonus: 0,
      streak: {
        enabled: false,
        maxStreakDays: 0,
        maxStreakMultiplier: 0,
      },
    },
    inbox: {
      enabled: false,
      maxMail: 0,
    },
    premium: {
      enabled: false,
    },
  },
  rateLimiting: {
    badAuthentication: {
      enabled: false,
      penalty: 0,
      flaggedStatusCodes: [],
    },
  },
  dailyLeaderboards: {
    enabled: false,
    maxResults: 0,
    leaderboardExpirationTimeInDays: 0,
    validModeRules: [],
    scheduleRewardsModeRules: [],
    topResultsToAnnounce: 1,
    xpRewardBrackets: [],
  },
  leaderboards: {
    minTimeTyping: 2 * 60 * 60,
    weeklyXp: {
      enabled: false,
      expirationTimeInDays: 0,
      xpRewardBrackets: [],
    },
  },
  connections: {
    enabled: false,
    maxPerUser: 100,
  },
};
