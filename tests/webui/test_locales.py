# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

import pytest

from llamafactory.webui.locales import ALERTS, LOCALES


# placeholders ({...}, %s), HTML tags and Markdown link targets must be kept as in English
FORMAT_PATTERN = re.compile(r"\{[^{}]*\}|%[sd]|<[^<>]+>|\]\([^()]+\)")


def _check_translation(name: str, en_text: str, tr_text: str | None) -> list[str]:
    if not isinstance(tr_text, str) or not tr_text.strip():
        return [f"{name}: missing or empty"]

    errors = []
    if sorted(FORMAT_PATTERN.findall(tr_text)) != sorted(FORMAT_PATTERN.findall(en_text)):
        errors.append(f"{name}: placeholders or markup differ from 'en'")

    # some messages are concatenated with other text at runtime, e.g. a file path or a link
    if tr_text.removeprefix(tr_text.rstrip()) != en_text.removeprefix(en_text.rstrip()):
        errors.append(f"{name}: trailing whitespace differs from 'en'")

    return errors


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_locales_turkish():
    errors = []
    for elem_name, entry in LOCALES.items():
        tr_entry = entry.get("tr")
        if not isinstance(tr_entry, dict) or set(tr_entry) != set(entry["en"]):
            errors.append(f"LOCALES[{elem_name}]: 'tr' keys must match 'en' keys {sorted(entry['en'])}")
            continue

        for key, en_text in entry["en"].items():
            errors.extend(_check_translation(f"LOCALES[{elem_name}][{key}]", en_text, tr_entry[key]))

    assert not errors, "\n".join(errors)


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_alerts_turkish():
    errors = []
    for alert_name, entry in ALERTS.items():
        errors.extend(_check_translation(f"ALERTS[{alert_name}]", entry["en"], entry.get("tr")))

    assert not errors, "\n".join(errors)
