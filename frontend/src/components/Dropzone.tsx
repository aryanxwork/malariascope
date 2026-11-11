import { useRef, useState, type DragEvent } from "react";
import { UploadCloud } from "lucide-react";

const ACCEPT = ["image/png", "image/jpeg", "image/bmp", "image/tiff", "image/webp"];

export default function Dropzone({ onFiles, disabled }: { onFiles: (files: File[]) => void; disabled?: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const accept = (list: FileList | null) => {
    if (!list) return;
    const files = Array.from(list).filter((f) => ACCEPT.includes(f.type) || /\.(png|jpe?g|bmp|tiff?|webp)$/i.test(f.name));
    if (files.length) onFiles(files);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    if (!disabled) accept(e.dataTransfer.files);
  };

  return (
    <div
      role="button"
      tabIndex={0}
      aria-disabled={disabled}
      onClick={() => !disabled && input.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && !disabled && input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-10 text-center transition ${
        over ? "border-accent bg-accent/5" : "border-line bg-panel hover:border-ink-3"
      } ${disabled ? "cursor-not-allowed opacity-60" : ""}`}
    >
      <span className="grid size-12 place-items-center rounded-full bg-accent/10 text-accent">
        <UploadCloud size={22} />
      </span>
      <div>
        <p className="font-medium">Drop single-cell images here, or click to browse</p>
        <p className="mt-1 text-sm text-ink-3">
          Segmented red-blood-cell crops from Giemsa-stained thin smears · PNG, JPG, BMP, TIFF, WEBP · up to 5 MB each
        </p>
      </div>
      <input
        ref={input}
        type="file"
        multiple
        accept={ACCEPT.join(",")}
        className="hidden"
        onChange={(e) => {
          accept(e.target.files);
          e.target.value = "";
        }}
      />
    </div>
  );
}
