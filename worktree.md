Пример для трёх Codex-задач через Git worktree.

Идея: для каждой Codex-задачи создаём отдельную рабочую папку и отдельную Git-ветку. Так Codex-агенты не мешают друг другу и не портят основную рабочую папку проекта.

Допустим, основной репозиторий находится здесь:

J:\Projects\Varus

Переходим в него:

cd /d J:\Projects\Varus

Проверяем текущие worktree:

git worktree list

Создаём три отдельные рабочие папки:

cd /d J:\Projects\Varus

git worktree add J:\Projects\Varus-agent-impl -b codex/impl
git worktree add J:\Projects\Varus-agent-tests -b codex/tests
git worktree add J:\Projects\Varus-agent-docs -b codex/docs

После этого появятся три отдельные папки:

J:\Projects\Varus-agent-impl -> ветка codex/impl
J:\Projects\Varus-agent-tests -> ветка codex/tests
J:\Projects\Varus-agent-docs -> ветка codex/docs

Codex Agent 1 — Implementation

Открыть в Rider / Codex папку:

J:\Projects\Varus-agent-impl

Задача для Codex:

Role: Implementation agent.

Task:
Implement the feature from the YouTrack ticket.

Rules:

Follow the existing architecture.
Do not rewrite unrelated code.
Add minimal necessary production code.
Do not update documentation except comments needed for code clarity.

Output:

Summary of changed files.
Tests that should be run.
Risks / assumptions.

Codex Agent 2 — Tests

Открыть в Rider / Codex папку:

J:\Projects\Varus-agent-tests

Задача для Codex:

Role: Test agent.

Task:
Add tests for the feature from the YouTrack ticket.

Rules:

Prefer unit/integration tests.
Do not change production code unless required to make code testable.
If production code needs changes, explain why.

Output:

Summary of added tests.
Commands to run.
Gaps not covered.

Codex Agent 3 — Docs

Открыть в Rider / Codex папку:

J:\Projects\Varus-agent-docs

Задача для Codex:

Role: Documentation agent.

Task:
Update project documentation for the implemented feature.

Rules:

Update README / CHANGELOG / relevant docs only if needed.
Do not change production code.
Do not change tests.
Keep documentation consistent with actual project behavior.

Output:

Summary of changed docs.
Any missing information that should be clarified.

Как потом слить результат обратно.

Например, если реализация в ветке codex/impl готова:

cd /d J:\Projects\Varus
git switch main
git merge codex/impl

Если тесты готовы:

git merge codex/tests

Если документация готова:

git merge codex/docs

Лучше перед merge проверить diff:

git diff main..codex/impl
git diff main..codex/tests
git diff main..codex/docs

Как удалить worktree после завершения:

cd /d J:\Projects\Varus

git worktree remove J:\Projects\Varus-agent-impl
git worktree remove J:\Projects\Varus-agent-tests
git worktree remove J:\Projects\Varus-agent-docs

Если Git ругается на незакоммиченные изменения, сначала зайти в нужную папку и проверить:

cd /d J:\Projects\Varus-agent-impl
git status

Если изменения не нужны:

git reset --hard
git clean -fd

Потом снова удалить worktree:

cd /d J:\Projects\Varus
git worktree remove J:\Projects\Varus-agent-impl

Короткое правило:

Один маленький тикет -> обычная branch
Один рискованный тикет -> worktree
Несколько Codex-задач параллельно -> несколько worktree

Worktree — это не замена branch. Это отдельная физическая папка проекта, привязанная к отдельной branch.