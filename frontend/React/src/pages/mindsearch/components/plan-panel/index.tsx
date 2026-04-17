import { useState } from 'react';
import styles from './index.module.less';

export interface IPlanStage {
    id: string;
    title: string;
    description: string;
    search_hints: string[];
    depends_on: string[];
    status: 'pending' | 'active' | 'done';
    summary?: string;
}

export interface IPlan {
    stages: IPlanStage[];
}

interface IProps {
    plan: IPlan;
    awaitingConfirmation: boolean;
    progressMessage?: string;
    onConfirm: () => void;
    onAmend: (text: string) => void;
    onCancel: () => void;
    disabled?: boolean;
}

const StatusDot = ({ status }: { status: string }) => {
    const cls = status === 'done'
        ? styles.dotDone
        : status === 'active'
            ? styles.dotActive
            : styles.dotPending;
    return <span className={`${styles.dot} ${cls}`} />;
};

const PlanPanel = ({
    plan, awaitingConfirmation, progressMessage,
    onConfirm, onAmend, onCancel, disabled,
}: IProps) => {
    const [amendText, setAmendText] = useState('');
    const [expanded, setExpanded] = useState<Record<string, boolean>>({});

    const toggle = (id: string) =>
        setExpanded(prev => ({ ...prev, [id]: !prev[id] }));

    const submitAmend = () => {
        const t = amendText.trim();
        if (!t) return;
        onAmend(t);
        setAmendText('');
    };

    return (
        <div className={styles.planPanel}>
            <div className={styles.header}>
                <div className={styles.title}>Research Plan</div>
                <div className={styles.sub}>
                    {plan.stages.length} stage{plan.stages.length === 1 ? '' : 's'}
                    {awaitingConfirmation ? ' — review and amend, or start research' : ''}
                </div>
            </div>

            {progressMessage && (
                <div className={styles.progress}>{progressMessage}</div>
            )}

            <ol className={styles.stageList}>
                {plan.stages.map(stage => {
                    const isOpen = !!expanded[stage.id];
                    return (
                        <li key={stage.id} className={styles.stageCard}>
                            <div className={styles.stageHead} onClick={() => toggle(stage.id)}>
                                <StatusDot status={stage.status} />
                                <span className={styles.stageId}>[{stage.id}]</span>
                                <span className={styles.stageTitle}>{stage.title}</span>
                                {stage.depends_on.length > 0 && (
                                    <span className={styles.deps}>
                                        ← {stage.depends_on.join(', ')}
                                    </span>
                                )}
                                <span className={styles.caret}>{isOpen ? '▾' : '▸'}</span>
                            </div>
                            {isOpen && (
                                <div className={styles.stageBody}>
                                    <div className={styles.desc}>{stage.description}</div>
                                    {stage.search_hints?.length > 0 && (
                                        <>
                                            <div className={styles.hintsLabel}>Search hints:</div>
                                            <ul className={styles.hintsList}>
                                                {stage.search_hints.map((h, i) => (
                                                    <li key={i}>{h}</li>
                                                ))}
                                            </ul>
                                        </>
                                    )}
                                    {stage.summary && (
                                        <>
                                            <div className={styles.hintsLabel}>Findings:</div>
                                            <div className={styles.summary}>
                                                {stage.summary.length > 600
                                                    ? stage.summary.slice(0, 600) + '…'
                                                    : stage.summary}
                                            </div>
                                        </>
                                    )}
                                </div>
                            )}
                        </li>
                    );
                })}
            </ol>

            {awaitingConfirmation && (
                <div className={styles.actions}>
                    <input
                        className={styles.amendInput}
                        value={amendText}
                        placeholder='Amend the plan (e.g. "add a stage on X", "drop s3", "make s4 independent")'
                        onChange={e => setAmendText(e.target.value)}
                        onKeyDown={e => { if (e.key === 'Enter') submitAmend(); }}
                        disabled={disabled}
                    />
                    <button
                        className={`${styles.btn} ${styles.btnSecondary}`}
                        onClick={submitAmend}
                        disabled={disabled || !amendText.trim()}
                    >
                        Apply amendment
                    </button>
                    <button
                        className={`${styles.btn} ${styles.btnPrimary}`}
                        onClick={onConfirm}
                        disabled={disabled}
                    >
                        Start research
                    </button>
                    <button
                        className={`${styles.btn} ${styles.btnGhost}`}
                        onClick={onCancel}
                        disabled={disabled}
                    >
                        Cancel
                    </button>
                </div>
            )}
        </div>
    );
};

export default PlanPanel;
