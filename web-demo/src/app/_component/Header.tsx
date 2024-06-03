export default function Header() {
  return (
    <header className="w-full max-w-12c flex items-end justify-between">
      <a
        href="/"
        className="border-4 border-solid border-black w-20 h-16 rounded-br-3xl flex items-center justify-center"
      >
        <h1 className="font-bold leading-4">
          YOUNG
          <br />
          YANG
          <br />
          GANG
        </h1>
      </a>
      <a href="/">
        <h1 className="text-3xl font-bold">multimodal7b demo</h1>
      </a>
    </header>
  );
}
