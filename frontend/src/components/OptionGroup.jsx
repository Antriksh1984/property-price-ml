/** Row of toggle buttons for small integer choices (BHK, bathrooms). */
export default function OptionGroup({ label, options, value, onChange, isDisabled = () => false, name }) {
  return (
    <div className="option-group" role="radiogroup" aria-label={label} style={{ "--count": options.length }}>
      {options.map((opt) => {
        const disabled = isDisabled(opt);
        return (
          <button
            key={opt}
            type="button"
            role="radio"
            name={name}
            aria-checked={value === opt}
            className={`option ${value === opt ? "is-selected" : ""}`}
            disabled={disabled}
            onClick={() => onChange(opt)}
          >
            {opt}
          </button>
        );
      })}
    </div>
  );
}
