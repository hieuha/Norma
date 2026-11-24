package harness

import (
	"context"
	"iter"
	"strings"
	"testing"
	"time"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// drainWithCtx is drain() over a caller-supplied ctx, so a test can cancel the
// parent (pause/kill/shutdown) independently of the run's own MaxDuration budget.
func drainWithCtx(t *testing.T, ctx context.Context, in QueryInput, deps QueryDeps) *Terminal {
	t.Helper()
	var term *Terminal
	for ev, err := range Query(ctx, in, deps) {
		if err != nil {
			t.Fatalf("event err: %v", err)
		}
		if ev.Kind == KindResult {
			term = ev.Terminal
		}
	}
	if term == nil {
		t.Fatal("no terminal event")
	}
	return term
}

// When a tool overruns the MaxDuration wall-clock budget, the tool is interrupted
// at the deadline and the run enters the WRAP-UP phase (settlement) on the live
// ctx — finishing with ReasonTimeout and the wrap-up prompt injected — instead of
// dying with ReasonAbortedTools. This is the hung/slow-tool case that the old
// turn-boundary-only budget check could never settle.
func TestSettlementOnToolOverrunsBudget(t *testing.T) {
	m := &scriptedModel{turns: [][]llm.StreamEvent{
		toolTurn("s1", "sleep", `{"seconds":3600}`), // hangs far past MaxDuration
		textTurn("已写回事实并总结"),                        // settlement turn → natural stop
	}}
	in := QueryInput{
		Messages:       []llm.Message{llm.UserText("go")},
		Tools:          tool.NewRegistry(tool.NewSleep()),
		PermissionMode: permission.ModeBypass,
		WorkingDir:     t.TempDir(),
		MaxDuration:    100 * time.Millisecond, // budget the run so the sleep overruns it
		Settlement:     &Settlement{Prompt: "SETTLE_ON_TIMEOUT 请写回事实"},
	}
	term := drainWithCtx(t, context.Background(), in, QueryDeps{CallModel: m.call})

	if term.Reason != ReasonTimeout {
		t.Fatalf("reason=%v, want ReasonTimeout（工具超墙→就地收尾，非 aborted_tools）", term.Reason)
	}
	found := false
	for _, msg := range term.Messages {
		for _, c := range msg.Content {
			if strings.Contains(c.Text, "SETTLE_ON_TIMEOUT") {
				found = true
			}
		}
	}
	if !found {
		t.Fatalf("超墙收尾提示未被注入到对话")
	}
	if m.calls != 2 {
		t.Fatalf("应有 2 次模型调用(主+收尾)，得 %d", m.calls)
	}
}

// Regression guard: a PARENT-ctx cancellation (pause / planner kill / shutdown) —
// as opposed to the run's own budget — still aborts with ReasonAbortedTools and
// does NOT enter the wrap-up phase.
func TestParentCancelStillAbortsNoSettlement(t *testing.T) {
	m := &scriptedModel{turns: [][]llm.StreamEvent{
		toolTurn("s1", "sleep", `{"seconds":3600}`),
		textTurn("would-be settlement"), // must NOT be reached
	}}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	in := QueryInput{
		Messages:       []llm.Message{llm.UserText("go")},
		Tools:          tool.NewRegistry(tool.NewSleep()),
		PermissionMode: permission.ModeBypass,
		WorkingDir:     t.TempDir(),
		// no MaxDuration: the only deadline is the parent ctx above
		Settlement: &Settlement{Prompt: "SHOULD_NOT_APPEAR"},
	}
	term := drainWithCtx(t, ctx, in, QueryDeps{CallModel: m.call})

	if term.Reason != ReasonAbortedTools {
		t.Fatalf("reason=%v, want ReasonAbortedTools（父 ctx 取消=真中断，不收尾）", term.Reason)
	}
	for _, msg := range term.Messages {
		for _, c := range msg.Content {
			if strings.Contains(c.Text, "SHOULD_NOT_APPEAR") {
				t.Fatalf("父 ctx 取消时不应进入收尾，却注入了收尾提示")
			}
		}
	}
	if m.calls != 1 {
		t.Fatalf("父取消后不应再调用模型收尾，模型调用数=%d，want 1", m.calls)
	}
}

// blockingModel blocks on the run ctx for a chosen set of calls (simulating a slow
// model turn that only returns when its ctx deadline fires), yielding the ctx error
// like a real provider whose request was cancelled; other calls replay preset events.
type blockingModel struct {
	calls int
	block []bool              // per call index: true = block on ctx then yield ctx.Err()
	turns [][]llm.StreamEvent // per call index: events when not blocking (nil when blocking)
}

func (m *blockingModel) call(ctx context.Context, _ llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	i := m.calls
	m.calls++
	return func(yield func(llm.StreamEvent, error) bool) {
		if i < len(m.block) && m.block[i] {
			<-ctx.Done()
			yield(llm.StreamEvent{}, ctx.Err())
			return
		}
		for _, ev := range m.turns[i] {
			if !yield(ev, nil) {
				return
			}
		}
	}
}

// A MODEL turn (not a tool) that overruns the MaxDuration budget is cut mid-stream
// by the run's own deadline and the run enters the WRAP-UP phase with ReasonTimeout
// — NOT ReasonModelError. This is the slow-model case (huge context + a reasoning
// model) the old ctx-based model call could never bound.
func TestSettlementOnModelTurnOverrunsBudget(t *testing.T) {
	m := &blockingModel{
		block: []bool{true, false},
		turns: [][]llm.StreamEvent{nil, textTurn("已写回并总结")},
	}
	in := QueryInput{
		Messages:       []llm.Message{llm.UserText("go")},
		Tools:          tool.NewRegistry(),
		PermissionMode: permission.ModeBypass,
		WorkingDir:     t.TempDir(),
		MaxDuration:    100 * time.Millisecond, // the first model turn blocks past this
		Settlement:     &Settlement{Prompt: "SETTLE_MODEL_TIMEOUT 请写回"},
	}
	term := drainWithCtx(t, context.Background(), in, QueryDeps{CallModel: m.call})

	if term.Reason != ReasonTimeout {
		t.Fatalf("reason=%v, want ReasonTimeout（模型调用超墙→收尾，非 model_error）", term.Reason)
	}
	found := false
	for _, msg := range term.Messages {
		for _, c := range msg.Content {
			if strings.Contains(c.Text, "SETTLE_MODEL_TIMEOUT") {
				found = true
			}
		}
	}
	if !found {
		t.Fatalf("模型超墙后未进入收尾（收尾提示未注入）")
	}
	if m.calls != 2 {
		t.Fatalf("应有 2 次模型调用(主+收尾)，得 %d", m.calls)
	}
}

// The WRAP-UP phase has its own wall-clock budget (Settlement.MaxDuration): if a
// wrap-up model turn itself overruns it, the run finishes with the settle reason
// instead of running settleMaxTurns unbounded. The test terminating at all proves
// the wrap-up was bounded — a regression would hang on the second blocked call.
func TestSettlementPhaseBoundedByGrace(t *testing.T) {
	m := &blockingModel{
		block: []bool{true, true}, // main turn blocks past budget; wrap-up turn blocks past grace
		turns: [][]llm.StreamEvent{nil, nil},
	}
	in := QueryInput{
		Messages:       []llm.Message{llm.UserText("go")},
		Tools:          tool.NewRegistry(),
		PermissionMode: permission.ModeBypass,
		WorkingDir:     t.TempDir(),
		MaxDuration:    50 * time.Millisecond,
		Settlement:     &Settlement{Prompt: "SETTLE", MaxDuration: 50 * time.Millisecond},
	}
	done := make(chan *Terminal, 1)
	go func() { done <- drainWithCtx(t, context.Background(), in, QueryDeps{CallModel: m.call}) }()
	select {
	case term := <-done:
		if term.Reason != ReasonTimeout {
			t.Fatalf("reason=%v, want ReasonTimeout（收尾阶段超其墙→以收尾原因结束）", term.Reason)
		}
		if m.calls != 2 {
			t.Fatalf("应为 2 次模型调用(主超墙+收尾超墙)，得 %d", m.calls)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("收尾阶段未被墙约束，运行挂死（回归）")
	}
}
