import styles from './index.module.less';
import IconFont from '../iconfont';

const Notice = () => {
    return <>
        <div className={styles.notice}>For a smoother experience, run this locally —-<a href='https://github.com/internLM/mindsearch' target='_blank'>Mind Search <IconFont type='icon-GithubFilled' /></a></div>
        <div className={styles.notice}>Powered by InternLM2.5, this service has been specifically optimized for Chinese. For a better experience in English, you can build it locally.</div>
    </>;
};
export default Notice;