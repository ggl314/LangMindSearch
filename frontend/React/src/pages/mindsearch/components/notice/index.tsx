import styles from './index.module.less';
import IconFont from '../iconfont';

const Notice = () => {
    return <>
        <div className={styles.notice}>For a smoother experience, run this locally —-<a href='https://github.com/internLM/mindsearch' target='_blank'>Mind Search <IconFont type='icon-GithubFilled' /></a></div>
    </>;
};
export default Notice;