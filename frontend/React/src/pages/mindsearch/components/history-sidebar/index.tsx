import { useState, useEffect, useCallback } from 'react';
import styles from './index.module.less';

interface HistoryItem {
    id: string;
    title: string;
    saved_at: string;
}

interface Props {
    qaList: any[];
    chatIsOver: boolean;
    onLoad: (qaList: any[]) => void;
}

const HistorySidebar = ({ qaList, chatIsOver, onLoad }: Props) => {
    const [items, setItems] = useState<HistoryItem[]>([]);
    const [selectedId, setSelectedId] = useState<string>('');
    const [titleInput, setTitleInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState('');

    const fetchList = useCallback(async () => {
        try {
            const res = await fetch('/history');
            if (res.ok) setItems(await res.json());
        } catch { /* ignore */ }
    }, []);

    useEffect(() => { fetchList(); }, [fetchList]);

    const formatDate = (isoStr: string) => {
        try {
            const d = new Date(isoStr);
            return d.toLocaleString(undefined, {
                month: 'short', day: 'numeric',
                hour: '2-digit', minute: '2-digit',
            });
        } catch { return isoStr; }
    };

    const handleMakeTitle = async () => {
        if (!qaList?.length) {
            setStatus('No research to title');
            return;
        }
        setLoading(true);
        setStatus('Generating title…');
        try {
            const res = await fetch('/history/make-title', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ data: qaList }),
            });
            const json = await res.json();
            if (json.title) {
                setTitleInput(json.title);
                setStatus('');
            } else {
                setStatus(json.error || 'Failed');
            }
        } catch (e: any) {
            setStatus(e.message);
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        if (!qaList?.length) {
            setStatus('No research to save');
            return;
        }
        let title = titleInput.trim();
        if (!title) {
            await handleMakeTitle();
            // titleInput state update is async; read fresh value after
            // We'll re-read from ref pattern — simpler: just require user to retry
            setStatus('Title generated — press Save again');
            return;
        }
        setLoading(true);
        setStatus('Saving…');
        try {
            const res = await fetch('/history', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title, data: qaList }),
            });
            if (res.ok) {
                setStatus('Saved');
                setTitleInput('');
                await fetchList();
                setTimeout(() => setStatus(''), 2000);
            } else {
                setStatus('Save failed');
            }
        } catch (e: any) {
            setStatus(e.message);
        } finally {
            setLoading(false);
        }
    };

    const handleLoad = async () => {
        if (!selectedId) {
            setStatus('Select a research first');
            return;
        }
        setLoading(true);
        setStatus('Loading…');
        try {
            const res = await fetch(`/history/${selectedId}`);
            if (res.ok) {
                const rec = await res.json();
                onLoad(rec.data);
                setStatus('Loaded');
                setTimeout(() => setStatus(''), 2000);
            } else {
                setStatus('Load failed');
            }
        } catch (e: any) {
            setStatus(e.message);
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        await fetch(`/history/${id}`, { method: 'DELETE' });
        if (selectedId === id) setSelectedId('');
        await fetchList();
    };

    return (
        <div className={styles.sidebar}>
            <div className={styles.header}>Saved Researches</div>
            <div className={styles.list}>
                {items.length === 0 && (
                    <div className={styles.empty}>No saved researches</div>
                )}
                {items.map(item => (
                    <div
                        key={item.id}
                        className={`${styles.item} ${selectedId === item.id ? styles.selected : ''}`}
                        onClick={() => setSelectedId(item.id)}
                    >
                        <div className={styles.itemTitle}>{item.title}</div>
                        <div className={styles.itemDate}>{formatDate(item.saved_at)}</div>
                        <button
                            className={styles.deleteBtn}
                            onClick={(e) => handleDelete(item.id, e)}
                            title="Delete"
                        >×</button>
                    </div>
                ))}
            </div>
            <div className={styles.controls}>
                <input
                    className={styles.titleInput}
                    type="text"
                    placeholder="Research title…"
                    value={titleInput}
                    onChange={e => setTitleInput(e.target.value)}
                    maxLength={80}
                />
                {status && <div className={styles.status}>{status}</div>}
                <div className={styles.buttons}>
                    <button
                        className={styles.btn}
                        onClick={handleSave}
                        disabled={loading || !chatIsOver}
                        title={!chatIsOver ? 'Wait for research to finish' : 'Save current research'}
                    >Save</button>
                    <button
                        className={styles.btn}
                        onClick={handleLoad}
                        disabled={loading || !selectedId}
                    >Load</button>
                    <button
                        className={styles.btn}
                        onClick={handleMakeTitle}
                        disabled={loading || !qaList?.length || !chatIsOver}
                        title="Generate title from research"
                    >Make Title</button>
                </div>
            </div>
        </div>
    );
};

export default HistorySidebar;
