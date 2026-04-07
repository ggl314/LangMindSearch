import styles from './index.module.less';
import { useEffect, useState, useRef, useMemo, useContext } from 'react';
import { Input, message, Tooltip } from 'antd';
import ShowRightIcon from './assets/think-progress-icon.svg';
import { MindsearchContext } from './provider/context';
import ChatRight from './components/chat-right';
import { useNavigate, useParams } from 'react-router-dom';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import SessionItem from './components/session-item';
import classNames from 'classnames';
import Notice from './components/notice';
import HistorySidebar from './components/history-sidebar';

interface INodeInfo {
    isEnd?: boolean; // 该节点是否结束
    current_node?: string;
    thinkingData?: string; // step1 思考
    queries?: [];
    readingData?: string; // step2 思考
    searchList?: [];
    conclusion?: string; // 节点的结论
    selectedIds?: number[];
    subQuestion?: string; // 节点的标题
    conclusionRef: any[];
    outputing?: boolean;
};

interface IFormattedData {
    question?: string;
    nodes?: any;
    adjacency_list?: object;
    response?: string;
    responseRefList?: any[];
    chatIsOver?: boolean;
};

interface INodeItem {
    id: string;
    name: string;
    state: number;
};

class FatalError extends Error { };
class RetriableError extends Error { };

const MindSearchCon = () => {
    const navigate = useNavigate();
    const params = useParams<{ id: string; robotId: string }>();
    const [qaList, setQaList] = useState<IFormattedData[]>([]);
    const [formatted, setFormatted] = useState<IFormattedData>({});
    const [question, setQuestion] = useState('');
    const [stashedQuestion, setStashedQuestion] = useState<string>('');
    const [newChatTip, setNewChatTip] = useState<Boolean>(false);
    const [singleObj, setSingleObj] = useState<any>(null);
    const [isEnd, setIsEnd] = useState(false);
    const [inputFocused, setFocused] = useState(false);
    // 一轮完整对话结束
    const [chatIsOver, setChatIsOver] = useState(true);

    const [currentNodeInfo, setCurrentNode] = useState<any>(null);
    const [currentNodeName, setCurrentNodeName] = useState<string>('');
    const [activeNode, setActiveNode] = useState<string>('');
    // 是否展示右侧内容
    const [showRight, setShowRight] = useState(false);
    const [adjList, setAdjList] = useState<any>({});
    const [historyNode, setHistoryNode] = useState<any>(null);

    const [hasNewChat, setHasNewChat] = useState(false);

    // Multi-turn graph state: persists across follow-up questions within a session.
    const [allNodes, setAllNodes] = useState<any[]>([]);
    const [originalQuestion, setOriginalQuestion] = useState<string>('');

    // Reconstruct the adjacency_list structure from a flat allNodes array.
    const buildAdjList = (nodes: any[]) => {
        if (!nodes?.length) return {};
        const seen: any = {};
        for (const n of nodes) seen[n.id] = n;
        const deduped: any[] = Object.values(seen);
        const adj: any = {};
        const rootChildren: any[] = [];
        deduped.forEach((node: any, i: number) => {
            const state = node.status === 'done' ? 3 : 1;
            rootChildren.push({ name: node.id, id: i + 1, state, depends_on: node.depends_on || [] });
            if (node.status === 'done') {
                adj[node.id] = [{ name: 'response', id: 100 + i, state: 0 }];
            }
        });
        adj['root'] = rootChildren;
        return adj;
    };

    // 新开会话
    const openNewChat = () => {
        location.reload();
    };

    const toggleRight = () => {
        setShowRight(!showRight);
    };

    // 渲染过程中保持渲染文字可见
    const keepScrollTop = () => {
        const divA = document.getElementById('chatArea') as HTMLDivElement;
        const divB = document.getElementById('messageWindowId') as HTMLDivElement;
        // 获取 divB 的当前高度
        const bHeight = divB.offsetHeight;

        // 检查 divA 是否需要滚动（即 divB 的高度是否大于 divA 的可视高度）
        if (bHeight > divA.offsetHeight) {
            // 滚动到 divB 的底部在 divA 的可视区域内  
            divA.scrollTop = bHeight - divA.offsetHeight + 30;
        }
    };

    const initPageState = () => {
        setSingleObj(null);
        setCurrentNodeName('');
        setCurrentNode(null);
        setFormatted({});
        // NOTE: adjList is intentionally NOT reset here so the graph persists
        // across follow-up questions. Only Clear and Load reset it.
        setShowRight(false);
        setIsEnd(false);
    };

    // data may be the new format {qaList, allNodes, originalQuestion}
    // or the legacy format (plain array = qaList only).
    const handleLoadHistory = (data: any) => {
        const isNewFormat = data && !Array.isArray(data);
        const loadedQaList: IFormattedData[] = isNewFormat ? (data.qaList || []) : data;
        const loadedAllNodes: any[] = isNewFormat ? (data.allNodes || []) : [];
        const loadedOriginalQ: string = isNewFormat
            ? (data.originalQuestion || loadedQaList[0]?.question || '')
            : (loadedQaList[0]?.question || '');

        console.log('[MS:load] format=%s qaList.length=%d allNodes.length=%d originalQuestion=%o',
            isNewFormat ? 'new' : 'legacy',
            loadedQaList.length,
            loadedAllNodes.length,
            loadedOriginalQ,
        );
        if (loadedAllNodes.length) {
            console.log('[MS:load] allNodes ids:', loadedAllNodes.map((n: any) => `${n.id}(${n.status})`).join(', '));
        } else {
            console.warn('[MS:load] allNodes is EMPTY (%s) — follow-ups will use originalQuestion=%o for context but planner will not see prior research summaries; re-save this session to enable full continuity',
                isNewFormat ? 'new format but no nodes saved' : 'legacy save format',
                loadedOriginalQ,
            );
        }

        initPageState();
        setAdjList(buildAdjList(loadedAllNodes));
        setQaList(loadedQaList);
        setAllNodes(loadedAllNodes);
        setOriginalQuestion(loadedOriginalQ);
        setChatIsOver(true);
        setCurrentNodeName('customer-0');
    };

    const handleStop = () => {
        ctrlRef.current?.abort();
        ctrlRef.current = null;
        setChatIsOver(true);
        initPageState();
        setCurrentNodeName('customer-0');
    };

    const handleClearResearch = () => {
        initPageState();
        setAdjList({});
        setQaList([]);
        setAllNodes([]);
        setOriginalQuestion('');
        setChatIsOver(true);
        setNewChatTip(false);
        setStashedQuestion('');
        localStorage.stashedNodes = '';
        localStorage.reformatStashedNodes = '';
    };

    const responseTimer: any = useRef(null);
    const ctrlRef = useRef<AbortController | null>(null);

    useEffect(() => {
        // console.log('[ms]---', formatted, chatIsOver, responseTimer.current);
        if (chatIsOver && formatted?.response) {
            // 一轮对话结束
            setQaList((pre) => {
                return pre.concat(formatted);
            });
            initPageState();
            setCurrentNodeName('customer-0');
        }
        if (!chatIsOver && !responseTimer.current) {
            responseTimer.current = setInterval(() => {
                keepScrollTop();
            }, 50);
        }
        if (responseTimer.current && chatIsOver) {
            // 如果 isEnd 变为 false，清除定时器  
            clearInterval(responseTimer.current);
            responseTimer.current = null;
        }
    }, [formatted?.response, chatIsOver, responseTimer.current, newChatTip]);

    useEffect(() => {
        if (formatted?.question) {
            setHistoryNode(null);
            setChatIsOver(false);
        }
    }, [formatted?.question]);

    // 存储节点信息
    const stashNodeInfo = (fullInfo: any, nodeName: string) => {
        // console.log('stash node info------', fullInfo, fullInfo?.response?.stream_state);
        const content = JSON.parse(fullInfo?.response?.content || '{}') || {};
        const searchListStashed: any = Object.keys(content).map((item) => {
            return { id: item, ...content[item] };
        });
        const stashedList = JSON.parse(localStorage?.stashedNodes || '{}');
        const nodeInfo = stashedList[nodeName] || {};

        if (fullInfo?.content) {
            nodeInfo.subQuestion = fullInfo.content;
        }
        if (fullInfo?.response?.formatted?.thought) {
            // step1 思考
            if (!nodeInfo?.readingData && !nodeInfo?.queries?.length) {
                nodeInfo.thinkingData = fullInfo?.response?.formatted?.thought;
            }

            // step2 思考
            if (nodeInfo?.thinkingData && nodeInfo?.queries?.length && nodeInfo?.searchList?.length && !nodeInfo?.selectedIds?.length && !nodeInfo?.conclusion) {
                nodeInfo.readingData = fullInfo?.response?.formatted?.thought;
            }

            // conclusion
            if (nodeInfo?.startConclusion && fullInfo?.response?.stream_state === 1) {
                nodeInfo.conclusion = fullInfo?.response?.formatted?.thought;
            }
        }
        if (fullInfo?.response?.formatted?.action?.parameters?.query?.length && !nodeInfo.queries?.length) {
            nodeInfo.queries = fullInfo?.response?.formatted.action.parameters.query;
        }

        if (searchListStashed?.length && !nodeInfo.conclusionRef) {
            nodeInfo.searchList = searchListStashed;
            nodeInfo.conclusionRef = content;
        }

        if (Array.isArray(fullInfo?.response?.formatted?.action?.parameters?.select_ids) && !nodeInfo?.selectedIds?.length) {
            nodeInfo.selectedIds = fullInfo?.response?.formatted.action.parameters.select_ids;
            nodeInfo.startConclusion = true;
        }

        if (fullInfo?.response?.stream_state) {
            nodeInfo.outputing = true;
        } else {
            nodeInfo.outputing = false;
        }

        const nodesList: any = {};
        nodesList[nodeName] = {
            current_node: nodeName,
            ...nodeInfo,
        };
        window.localStorage.stashedNodes = JSON.stringify({ ...stashedList, ...nodesList });
    };

    const formatData = (obj: any) => {
        // 嫦娥6号上有哪些国际科学载荷？它们的作用分别是什么？
        try {
            // 更新邻接表
            if (obj?.response?.formatted?.adjacency_list) {
                setAdjList(obj.response?.formatted?.adjacency_list);
            }

            if (!obj?.current_node && obj?.response?.formatted?.thought && obj?.response?.stream_state === 1) {
                // 有thought，没有node, planner思考过程
                setFormatted((pre: IFormattedData) => {
                    return {
                        ...pre,
                        response: obj.response.formatted.thought,
                    };
                });
            }
            if (obj?.response?.formatted?.ref2url && !formatted?.responseRefList) {
                setFormatted((pre: IFormattedData) => {
                    return {
                        ...pre,
                        responseRefList: obj?.response?.formatted?.ref2url,
                    };
                });
            }
            if (obj?.current_node || obj?.response?.formatted?.node) {
                // 有node, 临时存储node信息
                stashNodeInfo(obj?.response?.formatted?.node?.[obj.current_node], obj.current_node);
            }
        } catch (err) {
            console.log(err);
        }
    };

    const handleError = (errCode: number, msg: string) => {
        message.warning(msg || 'Request failed, please try again later');
        if (errCode === -20032 || errCode === -20033 || errCode === -20039) {
            // 敏感词校验失败, 新开会话
            openNewChat();
            return;
        }
        console.log('handle error------', msg);
        setChatIsOver(true);
        initPageState();
    };

    const startEventSource = () => {
        console.log('start event--------');
        if (qaList?.length > 4) {
            setNewChatTip(true);
            message.warning('Conversation limit reached, please start a new conversation');
            keepScrollTop();
            return;
        }
        setFormatted({ ...formatted, question });
        setQuestion('');
        setChatIsOver(false);
        const ctrl = new AbortController();
        ctrlRef.current = ctrl;
        const url = '/solve';

        // Multi-turn: if we have a root question already, this is a follow-up.
        // Send the accumulated node graph so the backend can extend it.
        const isFollowup = !!originalQuestion;
        const postData: any = { inputs: question };
        if (isFollowup) {
            postData.prior_nodes = allNodes;
            postData.original_question = originalQuestion;
            console.log('[MS:submit] FOLLOW-UP — original_question=%o prior_nodes.length=%d node_ids=%s',
                originalQuestion,
                allNodes.length,
                allNodes.map((n: any) => `${n.id}(${n.status})`).join(', ') || '(none)',
            );
        } else {
            // First question of this session — remember it as the root.
            setOriginalQuestion(question);
            console.log('[MS:submit] FIRST QUESTION — setting originalQuestion=%o', question);
        }

        fetchEventSource(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(postData),
            openWhenHidden: true,
            signal: ctrl.signal,
            onmessage(ev) {
                try {
                    const res = (ev?.data && JSON.parse(ev.data)) || null;
                    // nodes_snapshot: backend sends the full accumulated node list
                    // after finalize so we can persist it for the next follow-up.
                    if (res?.nodes_snapshot) {
                        console.log('[MS:sse] nodes_snapshot received: count=%d ids=%s',
                            res.nodes_snapshot.length,
                            res.nodes_snapshot.map((n: any) => `${n.id}(${n.status})`).join(', '),
                        );
                        setAllNodes(res.nodes_snapshot);
                        return;
                    }
                    if (res?.response?.stream_state === 0) {
                        setChatIsOver(true);
                        setFormatted((pre: IFormattedData) => {
                            return {
                                ...pre,
                                chatIsOver: true,
                            };
                        });
                    } else {
                        formatData(res);
                        setSingleObj(res);
                    }
                } catch (err) {
                    console.log('error on sse---', err);
                    handleError(0, 'Request failed, please try again later!');
                }
            },
            onerror(err) {
                console.log('error on sse---', err);
                handleError(0, '');
                ctrl.abort();
                throw err;
            },
            onclose() {
                // params?.id && handleUpdateHistoryItem(params?.id);
            }
        });
    };

    // 点击节点
    const handleNodeClick = (node: string, idx: number) => {
        if (isEnd && !chatIsOver) return; // 当节点输出完成，最终response进行中，不允许点击按钮，点击无效
        const isFromHistory = qaList?.[idx]?.nodes?.[node];
        setShowRight(true);
        setActiveNode(node);

        if (isFromHistory) {
            const info = qaList?.[idx]?.nodes?.[node];

            if (!info) {
                message.error('Could not read node information');
            }
            setHistoryNode(info);
        } else {
            setCurrentNodeName(node);
        }
    };

    // 解析历史记录或者搜索返回的数据
    const formatHistoryNode = (originNodeInfo: any) => {
        // console.log('format history node--------', originNodeInfo);
        const searchContent = JSON.parse(originNodeInfo?.memory?.[1]?.content || '{}') || {};
        const searchListStashed: any = Object.keys(searchContent).map((item) => {
            return { id: item, ...searchContent[item] };
        });

        const nodeInfo: INodeInfo = {
            current_node: originNodeInfo?.current_node || String(Date.now()),
            thinkingData: originNodeInfo?.memory?.[0]?.formatted?.thought || '', // step1 思考
            queries: originNodeInfo?.memory?.[0]?.formatted?.action?.parameters?.query || [],
            readingData: originNodeInfo?.memory?.[2]?.formatted?.thought || '', // step2 思考
            searchList: searchListStashed,
            conclusionRef: searchContent,
            conclusion: originNodeInfo?.memory?.[4]?.formatted?.thought || '', // 节点的结论
            selectedIds: originNodeInfo?.memory?.[2]?.formatted?.action?.parameters?.select_ids || [],
            subQuestion: originNodeInfo?.content, // 节点的标题
            isEnd: true,
            outputing: false
        };
        return nodeInfo;
    };

    const createSseChat = () => {
        if (submitDisabled) {
            return;
        }
        setQuestion(stashedQuestion);
        setStashedQuestion('');
        setCurrentNodeName('customer-0');
    };

    const checkNodesOutputFinish = () => {
        const adjListStr = JSON.stringify(adjList);
        // 服务端没有能准确描述所有节点输出完成的状态，前端从邻接表信息中寻找response信息，不保证完全准确，因为也可能不返回
        if (adjListStr.includes('"name":"response"')) {
            setIsEnd(true);
        }
    };

    useEffect(() => {
        if (!adjList) return;
        if (isEnd) {
            // 所有节点输出完成时收起右侧
            setShowRight(false);
        } else {
            checkNodesOutputFinish();
        }
        setFormatted((pre: IFormattedData) => {
            return {
                ...pre,
                adjacency_list: adjList,
            };
        });
    }, [adjList, isEnd]);

    useEffect(() => {
        const findStashNode = localStorage?.stashedNodes && JSON.parse(localStorage?.stashedNodes || '{}');
        if (!findStashNode || !currentNodeName) return;
        currentNodeName === 'customer-0' ? setCurrentNode(null) : setCurrentNode(findStashNode?.[currentNodeName]);
        currentNodeName !== 'customer-0' && setShowRight(true);
    }, [currentNodeName, localStorage?.stashedNodes]);

    useEffect(() => {
        if (!singleObj) return;
        if ((!currentNodeName || currentNodeName === 'customer-0') && singleObj?.current_node) {
            setCurrentNodeName(singleObj?.current_node);
        }
    }, [singleObj, currentNodeName]);

    useEffect(() => {
        if (question) {
            startEventSource();
        }
    }, [question]);

    useEffect(() => {
        if (!showRight) {
            setActiveNode('');
        }
    }, [showRight]);

    useEffect(() => {
        localStorage.stashedNodes = '';
        localStorage.reformatStashedNodes = '';

        return () => {
            // 返回清理函数，确保组件卸载时清除定时器  
            if (responseTimer.current) {
                clearInterval(responseTimer.current);
                responseTimer.current = null;
            }
        };
    }, []);

    const submitDisabled = useMemo(() => {
        return newChatTip || !stashedQuestion || !chatIsOver;
    }, [newChatTip, stashedQuestion, chatIsOver]);

    return (
        <MindsearchContext.Provider value={{
            isEnd,
            chatIsOver,
            activeNode: activeNode
        }}>
            <div className={styles.mainPage}>
                <HistorySidebar
                    qaList={qaList}
                    allNodes={allNodes}
                    originalQuestion={originalQuestion}
                    chatIsOver={chatIsOver}
                    onLoad={handleLoadHistory}
                />
                <div className={styles.chatContent}>
                    <div className={classNames(
                        styles.top,
                        (isEnd && !chatIsOver) ? styles.mb12 : ''
                    )} id="chatArea">
                        <div id="messageWindowId">
                            {qaList.length > 0 &&
                                qaList.map((item: IFormattedData, idx) => {
                                    return (
                                        <div key={`qa-item-${idx}`} className={styles.qaItem}>
                                            {
                                                item.question && <SessionItem
                                                    item={item}
                                                    handleNodeClick={handleNodeClick}
                                                    idx={idx}
                                                    key={`session-item-${idx}`}
                                                />
                                            }
                                        </div>
                                    );
                                })
                            }
                            {
                                formatted?.question &&
                                <SessionItem
                                    item={{ ...formatted, chatIsOver, isEnd, adjacency_list: adjList }}
                                    handleNodeClick={handleNodeClick}
                                    idx={qaList.length}
                                />
                            }
                        </div>
                        {newChatTip && (
                            <div className={styles.newChatTip}>
                                <span>
                                    Conversation limit reached, please start a <a onClick={openNewChat}>new conversation</a> 
                                </span>
                            </div>
                        )}
                    </div>
                    <div className={classNames(
                        styles.input,
                        inputFocused ? styles.focus : ''
                    )}>
                        <div className={styles.inputMain}>
                            <div className={styles.inputMainBox}>
                                <Input
                                    className={styles.textarea}
                                    variant="borderless"
                                    value={stashedQuestion}
                                    placeholder={'Start asking…'}
                                    onChange={(e) => {
                                        setStashedQuestion(e.target.value);
                                    }}
                                    onPressEnter={createSseChat}
                                    onFocus={() => { setFocused(true) }}
                                    onBlur={() => { setFocused(false) }}
                                />
                                <div className={classNames(styles.send, submitDisabled && styles.disabled)} onClick={createSseChat}>
                                    <i className="iconfont icon-Frame1" />
                                </div>
                            </div>
                            {!chatIsOver && (
                                <div className={styles.stopBtn} onClick={handleStop} title="Stop research">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 22 22" fill="currentColor">
                                        <rect x="4" y="4" width="14" height="14" rx="2" />
                                    </svg>
                                </div>
                            )}
                        </div>
                        {qaList.length > 0 && chatIsOver && (
                            <div className={styles.clearAction} onClick={handleClearResearch} title="Clear research">
                                Clear
                            </div>
                        )}
                    </div>
                    <Notice />
                </div>
                {showRight ? (
                    <ChatRight
                        nodeInfo={currentNodeInfo}
                        historyNode={historyNode}
                        toggleRight={toggleRight}
                        key={currentNodeName}
                        chatIsOver={chatIsOver}
                    />
                ) : (
                    <div className={styles.showRight}>
                        <div className={classNames(
                            styles.actionIcon,
                            isEnd && !chatIsOver ? styles.forbidden : ''
                        )} onClick={toggleRight}>
                            <Tooltip placement="leftTop" title="Thinking process">
                                <img src={ShowRightIcon} />
                            </Tooltip>
                        </div>
                    </div>
                )}
            </div>
        </MindsearchContext.Provider>
    );
};

export default MindSearchCon;
