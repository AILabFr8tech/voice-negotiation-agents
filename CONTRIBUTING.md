# Contributing

- **Bugs and questions:** open an issue with the command you ran, its full output,
  and your operating system and Python version.
- **Numbers in the paper:** every number is regenerated from the tables in
  [docs/report/data/](docs/report/data/) by the scripts in
  [shared/metrics/](shared/metrics/), and CI checks that they match the committed
  macros. A change that alters any regenerated macro must explain why in the pull
  request.
- **Spoken and model-facing Spanish:** the prompt, the tool descriptions and replies,
  the scenario lines and the word lists that match Spanish speech are part of the
  experiment. Do not translate or edit them in a pull request that is not about
  them.
- **Before a pull request:** run the test suites (see the README) and make sure the
  `reproduce` workflow passes.
- **License of contributions:** by opening a pull request you agree that your
  contribution is licensed under the [Apache License 2.0](LICENSE), the same
  license as the rest of the project (section 5 of the license).
