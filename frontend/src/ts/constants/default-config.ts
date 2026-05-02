import { Config, CustomThemeColors } from "@monkeytype/schemas/configs";
import { envConfig } from "virtual:env-config";

const obj: Config = {
  theme: "serika_dark",
  themeLight: "serika",
  themeDark: "serika_dark",
  autoSwitchTheme: false,
  customTheme: false,
  customThemeColors: [
    "#323437",
    "#e2b714",
    "#e2b714",
    "#646669",
    "#2c2e31",
    "#d1d0c5",
    "#ca4754",
    "#7e2a33",
    "#ca4754",
    "#7e2a33",
  ] as CustomThemeColors,
  favThemes: [],
  showKeyTips: true,
  smoothCaret: "medium",
  codeUnindentOnBackspace: false,
  quickRestart: "off",
  punctuation: false,
  numbers: false,
  words: 50,
  time: 30,
  mode: "time",
  quoteLength: [1],
  language: "english",
  fontSize: 2,
  freedomMode: false,
  difficulty: "normal",
  blindMode: false,
  quickEnd: false,
  caretStyle: "default",
  paceCaretStyle: "default",
  flipTestColors: false,
  layout: "default",
  funbox: [],
  confidenceMode: "off",
  indicateTypos: "off",
  compositionDisplay: "replace",
  timerStyle: "mini",
  liveSpeedStyle: "off",
  liveAccStyle: "off",
  liveBurstStyle: "off",
  colorfulMode: false,
  randomTheme: "off",
  timerColor: "main",
  timerOpacity: "1",
  stopOnError: "off",
  showAllLines: false,
  keymapMode: "off",
  keymapStyle: "staggered",
  keymapLegendStyle: "lowercase",
  keymapLayout: "overrideSync",
  keymapShowTopRow: "layout",
  keymapSize: 1,
  fontFamily: "Roboto_Mono",
  smoothLineScroll: false,
  alwaysShowDecimalPlaces: false,
  alwaysShowWordsHistory: false,
  singleListCommandLine: "on",
  capsLockWarning: true,
  playSoundOnError: "off",
  playSoundOnClick: "off",
  soundVolume: 0.5,
  startGraphsAtZero: true,
  showOutOfFocusWarning: true,
  paceCaret: "off",
  paceCaretCustomSpeed: 100,
  repeatedPace: true,
  accountChart: ["on", "on", "on", "on"],
  minWpm: "off",
  minWpmCustomSpeed: 100,
  highlightMode: "letter",
  typedEffect: "keep",
  typingSpeedUnit: "wpm",
  ads: "result",
  hideExtraLetters: false,
  strictSpace: false,
  minAcc: "off",
  minAccCustom: 90,
  monkey: false,
  repeatQuotes: "off",
  resultSaving: true,
  oppositeShiftMode: "off",
  customBackground: "",
  customBackgroundSize: "cover",
  customBackgroundFilter: [0, 1, 1, 1],
  customLayoutfluid: ["qwerty", "dvorak", "colemak"],
  customPolyglot: ["english", "spanish", "french", "german"],
  monkeyPowerLevel: "off",
  minBurst: "off",
  minBurstCustomSpeed: 100,
  burstHeatmap: false,
  britishEnglish: false,
  lazyMode: false,
  showAverage: "off",
  showPb: false,
  tapeMode: "off",
  tapeMargin: 50,
  maxLineWidth: 0,
  playTimeWarning: "off",
};

export function getDefaultConfig(): Config {
  const config = structuredClone(obj);

  if (envConfig.isTypeGptDemo) {
    config.theme = "chatgpt";
    config.themeDark = "chatgpt";
    config.themeLight = "chatgpt_light";
    config.mode = "words";
    config.words = 25;
    config.time = 30;
    config.language = "english_10k";
    config.funbox = ["llm"];
    config.showKeyTips = false;
    config.ads = "off";
    config.resultSaving = false;
    config.showOutOfFocusWarning = false;
  }

  return config;
}
