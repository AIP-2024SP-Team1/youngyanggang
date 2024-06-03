interface qna {
  question: string;
  answer: string;
}

interface IResponse {
  qnas: qna[];
}

export async function generate(context: string): Promise<IResponse> {
  const response = await fetch("http://localhost:3000/api", {
    method: "POST",
    cache: "no-cache",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ context }),
  });

  return response.json();
}
