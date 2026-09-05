// Redirects `import ... from "react"` to the UMD React already loaded as
// window.React by <script src=".../react.production.min.js">, so the
// esbuild bundle doesn't need a real "react" package installed.
export default window.React;
export const { useState, useRef, useEffect, useMemo, useCallback } = window.React;
