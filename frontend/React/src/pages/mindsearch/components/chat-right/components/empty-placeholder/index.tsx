import styles from './index.module.less';
import EmptyRightChatImg from '../../../../assets/empty-chat-right.svg';

const EmptyPlaceHolder = () => {
    return <>
        <div className={styles.emptyDiv}>
            <div className={styles.pic}>
                <img src={EmptyRightChatImg} />
            </div>
            <p>
                Select a node in the graph to view it here~
            </p>
        </div>
    </>
};

export default EmptyPlaceHolder;