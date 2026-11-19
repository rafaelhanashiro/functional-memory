// The Meetings page. Without a date in its address it lists the meetings, newest first, and
// adds one for a chosen date, optionally named. With ?date=YYYY-MM-DD it shows that meeting:
// its name and date, which "Edit meeting" changes (or deletes the meeting, after asking), and
// its agenda and notes (meeting/<date>-agenda.md and meeting/<date>-notes.md), each rendered,
// edited with the shared editor and saved through serve.py, and printed alone by its PDF button.
import {button, dirty, element, failure, saveText, showEditor, showView} from "./editor.js";

const API = "api/meetings";
const EMPTY = '<p class="empty">Nothing yet.</p>';
const DATE = /^\d{4}-\d{2}-\d{2}$/;
const LONG = {weekday: "long", month: "long", day: "numeric", year: "numeric"};
const SHORT = {weekday: "short", month: "short", day: "numeric", year: "numeric"};

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

// serve.py's meetings API: the reply's JSON, or an error with its message.
async function call(method, url, body) {
  const response = await fetch(url, body === undefined ? {method, cache: "no-store"} : {
    method, headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await failure(response));
  return response.status === 204 ? null : response.json();
}

// The list: a row per meeting, its date and name, under the form that adds one.

async function listMeetings() {
  const page = document.querySelector(".meetings");
  const form = page.querySelector(".new-meeting");
  const status = page.querySelector(".meeting-status");
  page.hidden = false;
  form.elements.date.value = today();
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const body = {date: form.elements.date.value, title: form.elements.title.value};
    for (const control of form.elements) control.disabled = true;
    try {
      const meeting = await call("POST", API, body);
      location.href = `meeting.html?date=${meeting.date}`;
    } catch (error) {
      status.textContent = error.message;
      for (const control of form.elements) control.disabled = false;
    }
  });
  try {
    const {meetings} = await call("GET", API);
    for (const meeting of meetings) {
      const row = element("a", "meeting-row");
      row.href = `meeting.html?date=${meeting.date}`;
      row.append(element("span", "meeting-row-date", spell(meeting.date, SHORT)));
      if (meeting.title) row.append(element("span", "meeting-row-title", meeting.title));
      page.querySelector(".meeting-list").append(row);
    }
    if (!meetings.length) status.textContent = "No meetings yet.";
  } catch {
    status.textContent = "Could not load the meetings.";
  }
}

// A meeting: its heading, the form that renames, moves or deletes it, and its two parts.

async function showMeeting(date) {
  const page = document.querySelector(".meeting");
  const heading = page.querySelector(".page-title");
  page.hidden = false;
  let meeting;
  try {
    meeting = DATE.test(date) && (await call("GET", API)).meetings.find((other) => other.date === date);
  } catch {
    heading.textContent = "Could not load this meeting";
    return;
  }
  if (!meeting) {
    heading.textContent = "No meeting on this date";
    return;
  }
  showHeading(page, meeting);
  editDetails(page, meeting);
  for (const part of page.querySelectorAll(".meeting-part")) {
    const sheet = part.querySelector(".sheet");
    const name = part.dataset.part;
    const path = `meeting/${date}-${name}.md`;
    const pdf = () => {
      const node = button("PDF", "pdf", () => savePdf(part, `${meeting.title || "Meeting"} ${name} ${meeting.date}`));
      node.title = "Save as PDF";
      return node;
    };
    const open = (text) => showView(sheet, text, {
      empty: EMPTY, onEdit: () => edit(text), actions: text.trim() ? [pdf()] : [],
    });
    const edit = (text) => showEditor(sheet, {
      key: sheet, text, empty: EMPTY, save: ({text: next}) => next === text ? undefined : saveText(path, next), close: open,
    });
    part.hidden = false;
    fetch(path, {cache: "no-store"})
      .then((response) => {
        if (!response.ok) throw new Error(response.statusText);
        return response.text();
      })
      .then(open)
      .catch(() => { sheet.innerHTML = '<p class="empty">Could not load this part.</p>'; });
  }
}

// The name heads the page, with the date under it; an unnamed meeting is headed by its date.
function showHeading(page, meeting) {
  const when = spell(meeting.date, LONG);
  const subtitle = page.querySelector(".meeting-date");
  page.querySelector(".page-title").textContent = meeting.title || when;
  subtitle.textContent = when;
  subtitle.hidden = !meeting.title;
  document.title = `${meeting.title || when} · Functional Memory`;
}

// "Edit meeting" swaps the heading for a form: the name, the date, and Delete. A new date
// renames the meeting's files, so the page reloads at its new address; the text being edited
// must be saved or cancelled first, since its old file would be gone.
function editDetails(page, meeting) {
  const head = page.querySelector(".meeting-head");
  const form = page.querySelector(".meeting-details");
  const status = form.querySelector(".meeting-form-status");
  const close = () => {
    form.hidden = true;
    head.hidden = false;
  };
  const edit = head.querySelector('[data-action="edit"]');
  edit.hidden = false;
  edit.addEventListener("click", () => {
    form.elements.title.value = meeting.title;
    form.elements.date.value = meeting.date;
    status.textContent = "";
    head.hidden = true;
    form.hidden = false;
    form.elements.title.focus();
  });
  form.querySelector('[data-action="cancel"]').addEventListener("click", close);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const changes = {};
    const title = form.elements.title.value.trim().replace(/\s+/g, " ");
    if (title !== meeting.title) changes.title = title;
    if (form.elements.date.value !== meeting.date) changes.date = form.elements.date.value;
    if (!Object.keys(changes).length) return close();
    if (changes.date && dirty.size) {
      status.textContent = "Save or cancel the agenda or notes you're editing first.";
      return;
    }
    try {
      const changed = await call("PATCH", `${API}/${meeting.date}`, changes);
      if (changed.date !== meeting.date) {
        location.replace(`meeting.html?date=${changed.date}`);
        return;
      }
      meeting.title = changed.title;
      showHeading(page, meeting);
      close();
    } catch (error) {
      status.textContent = error.message;
    }
  });
  form.querySelector('[data-action="delete"]').addEventListener("click", async () => {
    const what = meeting.title ? `"${meeting.title}" (${spell(meeting.date, LONG)})` : `the meeting on ${spell(meeting.date, LONG)}`;
    if (!confirm(`Delete ${what}, with its agenda and notes? This can't be undone.`)) return;
    try {
      await call("DELETE", `${API}/${meeting.date}`);
      dirty.clear();
      location.replace("meeting.html");
    } catch (error) {
      status.textContent = error.message;
    }
  });
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
