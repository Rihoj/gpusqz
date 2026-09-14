// Prints the version semantic-release would publish for HEAD, or an empty
// line when the commits since the last release don't call for one (or the
// branch isn't a release branch). CI runs it before building, so packages
// carry the version of the release they become part of. With $GITHUB_OUTPUT
// set it also writes version=<X.Y.Z or empty> and release=true|false there.
import { appendFileSync } from "node:fs";
import semanticRelease from "semantic-release";

// semantic-release logs to stdout; keep stdout for the answer alone.
const result = await semanticRelease({ dryRun: true }, { stdout: process.stderr, stderr: process.stderr });
const version = result && result.nextRelease ? result.nextRelease.version : "";
console.log(version);
if (process.env.GITHUB_OUTPUT) {
  appendFileSync(process.env.GITHUB_OUTPUT, `version=${version}\nrelease=${version ? "true" : "false"}\n`);
}
