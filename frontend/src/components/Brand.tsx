export function BrandMark({ light = false }: { light?: boolean }) {
  const c = light ? "#ffffff" : "#6d2177";
  const c2 = light ? "#e6c9ef" : "#a466b3";
  return (
    <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
      <path d="M16 4c3.2 4 3.8 9.6 0 15-3.8-5.4-3.2-11 0-15z" fill={c} />
      <path d="M5 11c5 0 9 3.4 10 9.5C9.4 20.2 5.6 16.6 5 11z" fill={c2} />
      <path d="M27 11c-5 0-9 3.4-10 9.5 5.6-.3 9.4-3.9 10-9.5z" fill={c2} />
      <path d="M7 23c3 2.6 6 3.6 9 3.6s6-1 9-3.6" fill="none" stroke={c} strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

export function Brand({ light = false, to = "/" }: { light?: boolean; to?: string }) {
  return (
    <a className="brand" href={to}>
      <BrandMark light={light} />
      <span className="brand-en">ATHAR</span>
      <span className="brand-ar" lang="ar">أثر</span>
    </a>
  );
}
