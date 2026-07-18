<script setup>
import { ref, computed, onMounted } from 'vue'
import Dialog from 'primevue/dialog'
import Toast from 'primevue/toast'
import { useToast } from 'primevue/usetoast'

const info = ref(null)
const samples = ref([])
const predictionsAvailable = ref(false)
const loading = ref(true)
const saving = ref(false)
const showCommit = ref(false)
const annotation = ref('')
const drafts = ref({})   // sample_id -> string currently being edited
const errors = ref({})   // sample_id -> parse error message
const toast = useToast()

const isClassification = computed(() => info.value?.task_type === 'classification')
const pendingCount = computed(() => samples.value.filter((s) => s.edited).length)

function toEditable(gold) {
  if (isClassification.value) return gold == null ? '' : String(gold)
  return gold == null ? '' : JSON.stringify(gold, null, 2)
}

function pretty(value) {
  if (value == null) return '—'
  return typeof value === 'string' ? value : JSON.stringify(value, null, 2)
}

async function load() {
  loading.value = true
  const [i, s] = await Promise.all([
    fetch('/api/info').then((r) => r.json()),
    fetch('/api/samples').then((r) => r.json()),
  ])
  info.value = i
  samples.value = s.samples
  predictionsAvailable.value = s.predictions_available
  const next = {}
  for (const smp of s.samples) next[smp.sample_id] = toEditable(smp.gold)
  drafts.value = next
  errors.value = {}
  loading.value = false
}

async function saveField(smp) {
  const raw = drafts.value[smp.sample_id]
  let gold
  try {
    if (isClassification.value) {
      gold = raw === '' ? null : raw
    } else {
      gold = raw.trim() === '' ? null : JSON.parse(raw)
    }
    errors.value[smp.sample_id] = null
  } catch (e) {
    errors.value[smp.sample_id] = 'Invalid JSON — not saved'
    return
  }
  const res = await fetch(`/api/samples/${encodeURIComponent(smp.sample_id)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ gold }),
  }).then((r) => r.json())
  smp.gold = res.gold
  smp.edited = res.edited
}

function useModelAnswer(smp) {
  drafts.value[smp.sample_id] = toEditable(smp.prediction)
  saveField(smp)
}

async function commit() {
  saving.value = true
  const res = await fetch('/api/session/commit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ annotation: annotation.value || null }),
  }).then((r) => r.json())
  saving.value = false
  showCommit.value = false
  annotation.value = ''
  if (res.committed) {
    toast.add({
      severity: 'success',
      summary: 'Session committed',
      detail: `${res.changed.length} sample(s) · ${String(res.commit).slice(0, 8)}`,
      life: 4000,
    })
    await load()
  } else {
    toast.add({ severity: 'info', summary: 'Nothing to commit', detail: 'No gold edits yet', life: 3000 })
  }
}

onMounted(load)
</script>

<template>
  <Toast />
  <div class="min-h-screen bg-background text-foreground">
    <!-- Header -->
    <header class="sticky top-0 z-10 border-b-2 border-border bg-secondary">
      <div class="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-5 py-4">
        <div>
          <h1 class="text-xl font-bold tracking-tight">
            Opaque · Relabeling
          </h1>
          <p v-if="info" class="font-mono text-sm text-muted-foreground">
            {{ info.project }} / {{ info.tool }} · {{ info.task_type }}
          </p>
        </div>
        <div class="flex items-center gap-3">
          <span class="border-2 border-border bg-card px-3 py-1 font-mono text-sm shadow-[4px_4px_0_0_var(--border)]">
            {{ pendingCount }} pending
          </span>
          <button
            class="border-2 border-border bg-primary px-4 py-2 font-semibold text-primary-foreground
                   shadow-[4px_4px_0_0_var(--border)] transition-transform
                   hover:-translate-x-[2px] hover:-translate-y-[2px]
                   disabled:cursor-not-allowed disabled:opacity-50"
            :disabled="pendingCount === 0"
            @click="showCommit = true"
          >
            Save session
          </button>
        </div>
      </div>
    </header>

    <main class="mx-auto max-w-5xl px-5 py-6">
      <p v-if="loading" class="font-mono text-muted-foreground">Loading…</p>

      <p
        v-else-if="!predictionsAvailable"
        class="mb-5 border-2 border-border bg-muted px-4 py-3 text-sm text-muted-foreground"
      >
        No prior run for the current prompt version — showing gold only. Run
        <code class="font-mono">opaque run</code> first to review against the model's predictions.
      </p>

      <div class="space-y-5">
        <article
          v-for="smp in samples"
          :key="smp.sample_id"
          class="border-2 border-border bg-card p-4 shadow-[4px_4px_0_0_var(--border)]"
        >
          <div class="mb-3 flex items-center justify-between">
            <div class="font-mono text-sm">
              <span class="font-bold">{{ smp.raw_file_name || smp.sample_id }}</span>
              <span class="text-muted-foreground"> · {{ smp.sample_id }}</span>
            </div>
            <span
              v-if="smp.edited"
              class="border-2 border-border bg-accent px-2 py-0.5 text-xs font-bold text-accent-foreground"
            >
              EDITED
            </span>
          </div>

          <div class="grid gap-4 md:grid-cols-2">
            <!-- Gold (editable) -->
            <div>
              <label class="mb-1 block text-xs font-bold uppercase tracking-wide">
                Gold (editable)
              </label>
              <input
                v-if="isClassification"
                v-model="drafts[smp.sample_id]"
                class="w-full border-2 border-input bg-background px-3 py-2 font-mono text-sm outline-none focus:ring-2 focus:ring-ring"
                @change="saveField(smp)"
              />
              <textarea
                v-else
                v-model="drafts[smp.sample_id]"
                rows="8"
                class="w-full resize-y border-2 border-input bg-background px-3 py-2 font-mono text-xs outline-none focus:ring-2 focus:ring-ring"
                @change="saveField(smp)"
              ></textarea>
              <p v-if="errors[smp.sample_id]" class="mt-1 text-xs font-bold text-primary">
                {{ errors[smp.sample_id] }}
              </p>
            </div>

            <!-- Model prediction (read-only) -->
            <div>
              <div class="mb-1 flex items-center justify-between">
                <label class="text-xs font-bold uppercase tracking-wide text-muted-foreground">
                  Model prediction
                </label>
                <button
                  v-if="smp.prediction != null"
                  class="border-2 border-border bg-secondary px-2 py-0.5 text-xs font-semibold
                         text-secondary-foreground hover:bg-accent hover:text-accent-foreground"
                  @click="useModelAnswer(smp)"
                >
                  Use this ↖
                </button>
              </div>
              <pre
                class="max-h-56 overflow-auto border-2 border-border bg-muted px-3 py-2 font-mono text-xs text-muted-foreground"
              >{{ pretty(smp.prediction) }}</pre>
            </div>
          </div>
        </article>
      </div>
    </main>

    <!-- Save-session dialog -->
    <Dialog v-model:visible="showCommit" modal header="Save relabeling session" :style="{ width: '30rem' }">
      <p class="mb-3 text-sm text-muted-foreground">
        Commits {{ pendingCount }} changed sample(s) as one commit in the project's git history.
      </p>
      <label class="mb-1 block text-xs font-bold uppercase tracking-wide">Note (optional)</label>
      <textarea
        v-model="annotation"
        rows="3"
        placeholder="e.g. fixed vendor names, filled missing dates"
        class="w-full border-2 border-input bg-background px-3 py-2 font-mono text-sm outline-none focus:ring-2 focus:ring-ring"
      ></textarea>
      <template #footer>
        <button class="border-2 border-border px-4 py-2 font-semibold" @click="showCommit = false">
          Cancel
        </button>
        <button
          class="border-2 border-border bg-primary px-4 py-2 font-semibold text-primary-foreground
                 shadow-[4px_4px_0_0_var(--border)] disabled:opacity-50"
          :disabled="saving"
          @click="commit"
        >
          {{ saving ? 'Committing…' : 'Commit session' }}
        </button>
      </template>
    </Dialog>
  </div>
</template>
