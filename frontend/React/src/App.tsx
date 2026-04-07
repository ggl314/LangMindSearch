import style from "./App.module.less";

import { BrowserRouter } from "react-router-dom";
import RouterRoutes from "@/routes/routes";
function App() {
  return (
    <BrowserRouter>
      <div className={style.app} id="app">
        <div className={style.header}>
          <div className={style.headerNav}>
            <div className={style.logoMark}>
              <div className={style.logoBars}>
                <span></span><span></span><span></span><span></span>
              </div>
            </div>
            <span className={style.logoText}>[Lang]MindSearch</span>
          </div>
        </div>

        <div className={style.content}>
          <RouterRoutes />
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;
