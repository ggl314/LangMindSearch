import { useEffect, useContext } from 'react';
import ReactFlow, {
    useNodesState,
    useEdgesState,
    MarkerType,
    Handle,
    Position,
    type Node,
    type Edge,
    type NodeProps,
} from 'reactflow';
import 'reactflow/dist/style.css';
import ELK from 'elkjs/lib/elk.bundled.js';
import classNames from 'classnames';
import { MindsearchContext } from '../../provider/context';
import styles from './index.module.less';

const elk = new ELK();

const NODE_W = 196;
const NODE_H = 38;

const EDGE_STYLE = { stroke: '#d7d8dd', strokeWidth: 1.5 };
const MARKER = { type: MarkerType.ArrowClosed, width: 14, height: 14, color: '#d7d8dd' };

// Custom node that matches the existing visual style
const SearchNodeInner = ({ data }: NodeProps) => {
    const { activeNode, chatIsOver, isEnd } = useContext(MindsearchContext);
    return (
        <>
            <Handle type="target" position={Position.Left} style={{ opacity: 0, pointerEvents: 'none' }} />
            <article
                className={classNames(
                    data.state === 1 ? styles.loading : data.state === 3 ? styles.finished : '',
                    data.isRoot || data.isFinal ? styles.init : '',
                    data.name === activeNode ? styles.active : '',
                    isEnd && !chatIsOver ? styles.forbidden : '',
                )}
                onClick={data.onClick}
            >
                <span title={data.label}>{data.label}</span>
                {data.state === 1 && <div className={styles.looping} />}
                {!data.isRoot && !data.isFinal && <div className={styles.finishDot} />}
                {data.name === activeNode && <div className={styles.dot} />}
            </article>
            <Handle type="source" position={Position.Right} style={{ opacity: 0, pointerEvents: 'none' }} />
        </>
    );
};

const nodeTypes = { searchNode: SearchNodeInner };

interface IProps {
    adjList: any;
    isEnd: boolean;
    handleNodeClick: (name: string) => void;
    listId: number;
}

const MindMapGraph = ({ adjList, isEnd, handleNodeClick }: IProps) => {
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);

    useEffect(() => {
        const rootItems: any[] = adjList?.root || [];
        if (!rootItems.length) return;

        const rfNodes: Node[] = [
            {
                id: '__root__',
                type: 'searchNode',
                data: { label: 'Original question', state: 3, isRoot: true, name: '__root__', onClick: undefined },
                position: { x: 0, y: 0 },
            },
        ];

        const rfEdges: Edge[] = [];

        for (const item of rootItems) {
            if (item.name === 'response') continue;

            rfNodes.push({
                id: item.name,
                type: 'searchNode',
                data: {
                    label: item.name,
                    state: item.state,
                    name: item.name,
                    onClick: item.state === 3 ? () => handleNodeClick(item.name) : undefined,
                },
                position: { x: 0, y: 0 },
            });

            const deps: string[] = item.depends_on || [];
            if (deps.length === 0) {
                rfEdges.push({
                    id: `__root__->${item.name}`,
                    source: '__root__',
                    target: item.name,
                    style: EDGE_STYLE,
                    markerEnd: MARKER,
                });
            } else {
                for (const dep of deps) {
                    rfEdges.push({
                        id: `${dep}->${item.name}`,
                        source: dep,
                        target: item.name,
                        style: EDGE_STYLE,
                        markerEnd: MARKER,
                    });
                }
            }
        }

        if (isEnd) {
            rfNodes.push({
                id: '__final__',
                type: 'searchNode',
                data: { label: 'Final response', state: 3, isFinal: true, name: '__final__', onClick: undefined },
                position: { x: 0, y: 0 },
            });
            for (const item of rootItems) {
                if (item.name !== 'response' && adjList[item.name]) {
                    rfEdges.push({
                        id: `${item.name}->__final__`,
                        source: item.name,
                        target: '__final__',
                        style: EDGE_STYLE,
                        markerEnd: MARKER,
                    });
                }
            }
        }

        const elkGraph = {
            id: 'root',
            layoutOptions: {
                'elk.algorithm': 'layered',
                'elk.direction': 'RIGHT',
                'elk.spacing.nodeNode': '20',
                'elk.layered.spacing.nodeNodeBetweenLayers': '60',
                'elk.layered.nodePlacement.strategy': 'SIMPLE',
                'elk.padding': '[top=20, left=20, bottom=20, right=20]',
            },
            children: rfNodes.map(n => ({ id: n.id, width: NODE_W, height: NODE_H })),
            edges: rfEdges.map(e => ({ id: e.id, sources: [e.source], targets: [e.target] })),
        };

        elk.layout(elkGraph).then(layout => {
            const positioned = rfNodes.map(node => {
                const en = layout.children?.find(n => n.id === node.id);
                return { ...node, position: { x: en?.x ?? 0, y: en?.y ?? 0 } };
            });
            setNodes(positioned);
            setEdges(rfEdges);
        }).catch(console.error);
    }, [adjList, isEnd]);

    return (
        <div className={styles.flowContainer}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                nodeTypes={nodeTypes}
                fitView
                fitViewOptions={{ padding: 0.15 }}
                nodesDraggable={false}
                nodesConnectable={false}
                elementsSelectable={false}
                zoomOnScroll={false}
                panOnScroll={false}
                panOnDrag={false}
                zoomOnPinch={false}
                zoomOnDoubleClick={false}
                proOptions={{ hideAttribution: true }}
            />
        </div>
    );
};

export default MindMapGraph;
