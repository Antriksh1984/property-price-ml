import { useEffect, useId, useMemo, useRef, useState } from "react";

const MAX_RESULTS = 60;

/**
 * Searchable combobox: type to filter, arrow keys to move, Enter to pick, Esc to close.
 * `value` is the committed location; the text box can hold a partial query.
 */
export default function LocationSelect({ locations, value, onChange, error, disabled }) {
  const [query, setQuery] = useState(value);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const listRef = useRef(null);
  const wrapRef = useRef(null);
  const id = useId();

  useEffect(() => setQuery(value), [value]);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    // Show everything when the box is empty or still shows the committed value.
    const list = !q || query === value ? locations : locations.filter((l) => l.toLowerCase().includes(q));
    // Prefix matches first, then substring matches.
    return [...list]
      .sort((a, b) => Number(!a.toLowerCase().startsWith(q)) - Number(!b.toLowerCase().startsWith(q)))
      .slice(0, MAX_RESULTS);
  }, [query, value, locations]);

  // Close when clicking outside; restore the committed value.
  useEffect(() => {
    const onDown = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) {
        setOpen(false);
        setQuery(value);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [value]);

  // Keep the highlighted option scrolled into view.
  useEffect(() => {
    listRef.current?.children[active]?.scrollIntoView({ block: "nearest" });
  }, [active]);

  const choose = (loc) => {
    onChange(loc);
    setQuery(loc);
    setOpen(false);
  };

  const onKeyDown = (e) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setOpen(true);
      setActive((i) => Math.min(i + 1, matches.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && open) {
      e.preventDefault();
      if (matches[active]) choose(matches[active]);
    } else if (e.key === "Escape") {
      setOpen(false);
      setQuery(value);
    }
  };

  const listId = `${id}-list`;
  return (
    <div className="combo" ref={wrapRef}>
      <div className={`combo-input ${error ? "has-error" : ""}`}>
        <svg aria-hidden="true" viewBox="0 0 20 20" className="combo-icon">
          <path d="M8.5 3a5.5 5.5 0 014.38 8.83l3.65 3.64-1.06 1.06-3.64-3.65A5.5 5.5 0 118.5 3zm0 1.5a4 4 0 100 8 4 4 0 000-8z" />
        </svg>
        <input
          id="location"
          type="text"
          role="combobox"
          aria-expanded={open}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={open && matches[active] ? `${id}-opt-${active}` : undefined}
          aria-invalid={Boolean(error)}
          placeholder={disabled ? "Loading localities…" : "Search a locality, e.g. Whitefield"}
          value={query}
          disabled={disabled}
          autoComplete="off"
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
            setActive(0);
            if (e.target.value === "") onChange("");
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={onKeyDown}
        />
      </div>
      {open && !disabled && (
        <ul className="combo-list" role="listbox" id={listId} ref={listRef}>
          {matches.length === 0 ? (
            <li className="combo-empty">No locality matches “{query}”. Try “other”.</li>
          ) : (
            matches.map((loc, i) => (
              <li
                key={loc}
                id={`${id}-opt-${i}`}
                role="option"
                aria-selected={loc === value}
                className={`combo-option ${i === active ? "is-active" : ""} ${loc === value ? "is-selected" : ""}`}
                onMouseDown={(e) => e.preventDefault()}
                onMouseEnter={() => setActive(i)}
                onClick={() => choose(loc)}
              >
                {loc === "other" ? "Other / not listed" : loc}
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  );
}
