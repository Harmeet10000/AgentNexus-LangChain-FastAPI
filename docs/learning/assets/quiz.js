/* ============================================================
   quiz.js — reusable self-checking quiz widget
   Usage:
     <div id="quiz1" class="quiz"></div>
     <script src="../assets/quiz.js"></script>
     <script>
       Quiz.render("quiz1", {
         questions: [
           {
             prompt: "…",
             options: ["…", "…", "…"],   // keep options equal-length!
             correct: 0,                  // index of right answer
             explain: "why — shown after answering"
           }
         ]
       });
     </script>
   ============================================================ */
(function () {
  "use strict";

  function renderQuestion(container, q, qi, state) {
    const block = document.createElement("div");
    block.className = "quiz__q";

    const prompt = document.createElement("div");
    prompt.className = "quiz__prompt";
    prompt.textContent = (qi + 1) + ". " + q.prompt;
    block.appendChild(prompt);

    const list = document.createElement("ul");
    list.className = "quiz__options";

    q.options.forEach((opt, oi) => {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.className = "quiz__option";
      btn.type = "button";
      btn.textContent = opt;

      btn.addEventListener("click", function () {
        if (state.answered[qi]) return;
        state.answered[qi] = true;
        state.score += oi === q.correct ? 1 : 0;

        Array.from(list.children).forEach((otherLi, otherOi) => {
          const b = otherLi.firstChild;
          b.disabled = true;
          if (otherOi === q.correct) b.classList.add("quiz__option--correct");
          else if (otherOi === oi) b.classList.add("quiz__option--wrong");
        });

        if (q.explain) {
          const ex = document.createElement("div");
          ex.className = "quiz__explain";
          ex.style.display = "block";
          ex.textContent = q.explain;
          block.appendChild(ex);
        }

        state.done += 1;
        if (state.done === state.questions.length) renderScore(container, state);
      });

      li.appendChild(btn);
      list.appendChild(li);
    });

    block.appendChild(list);
    container.appendChild(block);
  }

  function renderScore(container, state) {
    const scoreEl = document.createElement("div");
    scoreEl.className = "quiz__score";
    scoreEl.textContent =
      "Score: " + state.score + " / " + state.questions.length +
      (state.score === state.questions.length
        ? " — perfect. This one's moving to storage strength."
        : " — revisit the linked reference, then retry below.");

    const retry = document.createElement("button");
    retry.className = "quiz__retry";
    retry.type = "button";
    retry.textContent = "Retry quiz";

    retry.addEventListener("click", function () {
      const qs = state.questions;
      container.innerHTML = "";
      Quiz.render(container.id, { questions: qs });
      container.scrollIntoView({ behavior: "smooth", block: "start" });
    });

    scoreEl.appendChild(document.createElement("br"));
    scoreEl.appendChild(retry);
    container.appendChild(scoreEl);
  }

  window.Quiz = {
    render: function (containerId, config) {
      const container = document.getElementById(containerId);
      if (!container) return;
      container.innerHTML = "";

      const title = document.createElement("h3");
      title.textContent = "Check yourself";
      container.appendChild(title);

      const state = {
        questions: config.questions,
        answered: {},
        done: 0,
        score: 0
      };

      config.questions.forEach((q, qi) => renderQuestion(container, q, qi, state));
    }
  };
})();
