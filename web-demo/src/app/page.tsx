import Header from "./_component/Header";

export default function Home() {
  return (
    <>
      <section className="flex-1 max-w-4c p-8 flex flex-col gap-16 bg-white">
        <div className="flex">
          <h2 className="font-bold">context</h2>
        </div>
      </section>
      <div className="flex-[3] max-w-8c mx-auto flex flex-col">
        <Header />
        <main>home</main>
      </div>
    </>
  );
}
