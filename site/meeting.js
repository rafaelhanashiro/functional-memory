// The meeting agenda and notes: each a Markdown file (meeting/agenda.md, meeting/notes.md)
// shown rendered, edited with the shared editor, and saved through serve.py.
import {saveText, showEditor, showView} from "./editor.js";

const EMPTY = '<p class="empty">Nothing yet.</p>';

for (const part of document.querySelectorAll(".meeting-part")) {
  const sheet = part.querySelector(".sheet");
  const path = part.dataset.document;
  const open = (text) => showView(sheet, text, {empty: EMPTY, onEdit: () => edit(text)});
  const edit = (text) => showEditor(sheet, {
    key: sheet, text, empty: EMPTY, save: ({text: next}) => next === text ? undefined : saveText(path, next), close: open,
  });
  fetch(path, {cache: "no-store"})
    .then((response) => {
      if (!response.ok) throw new Error(response.statusText);
      return response.text();
    })
    .then(open)
    .catch(() => { sheet.innerHTML = '<p class="empty">Could not load this part.</p>'; });
}
