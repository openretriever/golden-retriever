// @ts-check
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';
import starlightThemeNova from 'starlight-theme-nova';

export default defineConfig({
  site: 'https://retriever-space.pages.dev',
  integrations: [
    starlight({
      title: 'Golden Retriever Reference',
      description: 'Applied robot examples, type packs, and Hub-pack candidates for Retriever.',
      logo: {
        src: './src/assets/retriever-illustrative.jpeg',
        alt: 'Golden Retriever',
      },
      favicon: '/assets/logo.svg',
      customCss: ['./src/styles/golden.css'],
      plugins: [starlightThemeNova()],
      sidebar: [
        {
          label: 'Start',
          items: [
            { label: 'Overview', link: '/' },
            { label: 'First Golden Proof', slug: 'examples/golden-hub-proof' },
            { label: 'Example Catalog', slug: 'examples' },
          ],
        },
        {
          label: 'Examples',
          items: [
            { label: 'Perception and Memory', slug: 'examples/perception-memory' },
            { label: 'Language and Grounding', slug: 'examples/language-grounding' },
            { label: 'Pipeline Composition', slug: 'examples/pipeline-composition' },
            { label: 'Simulation and Visualization', slug: 'examples/simulation-visualization' },
          ],
        },
        {
          label: 'Hub Packs',
          items: [
            { label: 'Golden Pack Boundary', slug: 'hub' },
            { label: 'Current Export Catalog', slug: 'hub/export-catalog' },
            { label: 'Pack Maturity Guide', slug: 'hub/pack-roadmap' },
          ],
        },
        {
          label: 'Robot Type Packs',
          items: [
            { label: 'Overview', slug: 'robot-type-packs' },
            { label: 'Type Catalog', slug: 'robot-type-packs/type-catalog' },
            { label: 'Flow Contracts', slug: 'robot-type-packs/flow-contracts' },
            { label: 'Data and Event Streams', slug: 'robot-type-packs/data-event-streams' },
            { label: 'LeRobot Dataset Export', slug: 'robot-type-packs/lerobot-export' },
          ],
        },
      ],
    }),
  ],
});
