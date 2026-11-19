// The Meetings page. Without a date in its address it lists the meetings, newest first, and
// adds one for a chosen date; with ?date=YYYY-MM-DD it shows that meeting's agenda and notes
// (meeting/<date>-agenda.md and meeting/<date>-notes.md), each rendered, edited with the
// shared editor and saved through serve.py, and printed on its own by its PDF button.
import {button, element, failure, saveText, showEditor, showView} from "./editor.js";

const EMPTY = '<p class="empty">Nothing yet.</p>';
const ARROW = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10h12M11 5l5 5-5 5"/></svg>';
const DATE = /^\d{4}-\d{2}-\d{2}$/;

// A YYYY-MM-DD date in words, read as a local date: new Date("2026-11-19") would be midnight
// UTC, which is still the evening before in America.
function spell(date, options) {
  const [year, month, day] = date.split("-").map(Number);
  return new Date(year, month - 1, day).toLocaleDateString("en-US", options);
}

function today() {
  const now = new Date();
  return [now.getFullYear(), now.getMonth() + 1, now.getDate()].map((part) => String(part).padStart(2, "0")).join("-");
}

// The list: a banner per meeting, as on the Runs page, under the form that adds one.

async function listMeetings() {
  const page = document.querySelector(".meetings");
  const form = page.querySelector(".new-meeting");
  const status = page.querySelector(".meeting-status");
  page.hidden = false;
  form.elements.date.value = today();
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const date = form.elements.date.value;
    for (const control of form.elements) control.disabled = true;
    try {
      const response = await fetch("api/meetings", {
        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({date}),
      });
      if (!response.ok) throw new Error(await failure(response));
      location.href = `meeting.html?date=${date}`;
    } catch (error) {
      status.textContent = error.message;
      for (const control of form.elements) control.disabled = false;
    }
  });
  try {
    const response = await fetch("api/meetings", {cache: "no-store"});
    if (!response.ok) throw new Error(response.statusText);
    const {dates} = await response.json();
    for (const date of dates) {
      const link = element("a", "subpage");
      link.href = `meeting.html?date=${date}`;
      const head = element("span", "subpage-head");
      head.append(element("span", "subpage-title", spell(date, {month: "long", day: "numeric", year: "numeric"})),
        element("span", "subpage-meta", spell(date, {weekday: "long"})));
      const arrow = element("span", "subpage-arrow");
      arrow.innerHTML = ARROW;
      link.append(head, arrow);
      page.querySelector(".subpages").append(link);
    }
    if (!dates.length) status.textContent = "No meetings yet.";
  } catch {
    status.textContent = "Could not load the meetings.";
  }
}

// A meeting: its agenda and notes, each a sheet with an Edit button and, once it has text, a
// PDF button.

function showMeeting(date) {
  const page = document.querySelector(".meeting");
  const heading = page.querySelector(".page-title");
  page.hidden = false;
  if (!DATE.test(date)) {
    heading.textContent = "No meeting on this date";
    for (const part of page.querySelectorAll(".meeting-part")) part.hidden = true;
    return;
  }
  heading.textContent = spell(date, {weekday: "long", month: "long", day: "numeric", year: "numeric"});
  document.title = `${heading.textContent} · Functional Memory`;
  for (const part of page.querySelectorAll(".meeting-part")) {
    const sheet = part.querySelector(".sheet");
    const name = part.dataset.part;
    const path = `meeting/${date}-${name}.md`;
    const pdf = () => {
      const node = button("PDF", "pdf", () => savePdf(part, `Meeting ${name} ${date}`));
      node.title = "Save as PDF";
      return node;
    };
    const open = (text) => showView(sheet, text, {
      empty: EMPTY, onEdit: () => edit(text), actions: text.trim() ? [pdf()] : [],
    });
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
}

// The PDF button prints its part alone, for the print dialog's Save as PDF. The page's title
// names the file, so while the dialog is open it says which part of which meeting it is.
function savePdf(part, name) {
  const title = document.title;
  document.title = name;
  document.body.classList.add("printing-part");
  part.classList.add("printing");
  window.addEventListener("afterprint", () => {
    document.title = title;
    document.body.classList.remove("printing-part");
    part.classList.remove("printing");
  }, {once: true});
  window.print();
}

const date = new URLSearchParams(location.search).get("date");
if (date === null) listMeetings();
else showMeeting(date);
