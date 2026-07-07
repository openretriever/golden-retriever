// @ts-check
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';
import starlightThemeNova from 'starlight-theme-nova';

export default defineConfig({
  site: 'https://retriever-space.pages.dev',
  integrations: [
    starlight({
      title: 'GoldenRetriever Reference',
      description: 'Applied Retriever examples, robot payload references, visualization lanes, and Hub pack candidates.',
      logo: {
        src: './src/assets/retriever-illustrative.jpeg',
        alt: 'GoldenRetriever',
      },
      favicon: '/assets/logo.svg',
      customCss: ['./src/styles/golden.css'],
      plugins: [starlightThemeNova()],
      sidebar: [
        {
          label: 'Start',
          items: [
            { label: 'Overview', link: '/' },
            { label: 'First GoldenRetriever Proof', slug: 'examples/golden-hub-proof' },
            { label: 'Example Catalog', slug: 'examples' },
          ],
        },
        {
          label: 'Run Examples',
          items: [
            { label: 'Perception and Memory', slug: 'examples/perception-memory' },
            { label: 'Language and Grounding', slug: 'examples/language-grounding' },
            { label: 'Pipeline Composition', slug: 'examples/pipeline-composition' },
            { label: 'Simulation and Visualization', slug: 'examples/simulation-visualization' },
          ],
        },
        {
          label: 'Notebooks',
          items: [
            { label: 'Language → Plan', slug: 'notebooks/language-caption-plan' },
          ],
        },
        {
          label: 'Reuse Robot Payloads',
          items: [
            { label: 'Payload Overview', slug: 'robot-payloads' },
            { label: 'Choose a Payload', slug: 'robot-payloads/type-catalog' },
            { label: 'Flow I/O Contracts', slug: 'robot-payloads/flow-contracts' },
            { label: 'Data and Event Streams', slug: 'robot-payloads/data-event-streams' },
            { label: 'LeRobot Export', slug: 'robot-payloads/lerobot-export' },
          ],
        },
        {
          label: 'Promote to Hub Packs',
          items: [
            { label: 'Pack Boundary', slug: 'hub' },
            { label: 'Export Catalog', slug: 'hub/export-catalog' },
            { label: 'Maturity Guide', slug: 'hub/pack-roadmap' },
          ],
        },
      ],
    }),
  ],
});
